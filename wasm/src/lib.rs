use glam::Vec3Swizzles;
use glam::{DMat3, DVec2, DVec3};
use ordered_float::OrderedFloat;
use wasm_bindgen::prelude::*;

const PI: f64 = std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PortalType {
    Flat,
    Semicircle { scale_y: f64 },
    Circle { scale_y: f64 },
    Perspective { scale_y: f64 },
    Wormhole { scale_y: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Portal {
    pub kind: PortalType,
    pub portal1: DMat3,
    pub portal2: DMat3,
    pub worlds: (usize, usize),

    pub portal1_inv: DMat3,
    pub portal2_inv: DMat3,
}

#[derive(Clone)]
pub struct RayInner {
    pub o: DVec2,
    pub d: DVec2,
}

impl RayInner {
    pub fn new(o: DVec2, d: DVec2) -> RayInner {
        RayInner { o, d }
    }

    pub fn offset(&self, t: f64) -> DVec2 {
        self.o + self.d * t
    }

    pub fn normalize(&self) -> RayInner {
        RayInner {
            o: self.o,
            d: self.d.normalize(),
        }
    }

    pub fn transform(&self, matrix: &DMat3) -> RayInner {
        RayInner {
            o: matrix.transform_point2(self.o),
            d: matrix.transform_vector2(self.d),
        }
    }
}

fn intersect_ellipse(ray: &RayInner, scale_y: f64) -> Option<f64> {
    if scale_y <= 0.0 {
        return None;
    }

    let o = DVec2::new(ray.o.x, ray.o.y / scale_y);
    let d = DVec2::new(ray.d.x, ray.d.y / scale_y);

    let a = d.dot(d);
    if a <= 1e-24 {
        return None;
    }
    let b = 2.0 * o.dot(d);
    let c = o.dot(o) - 1.0;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();

    let q = -0.5 * (b + b.signum() * sqrt_disc);

    let mut t0 = q / a;
    let mut t1 = c / q;
    if t0 > t1 {
        std::mem::swap(&mut t0, &mut t1);
    }

    const EPS: f64 = 1e-12;
    if t0 >= EPS {
        Some(t0)
    } else if t1 >= EPS {
        Some(t1)
    } else {
        None
    }
}

fn ray_circle_intersection(ray: &RayInner) -> Option<f64> {
    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.o.dot(ray.d);
    let c = ray.o.dot(ray.o) - 1.0;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    if t1 >= 0.0 && t2 >= 0.0 {
        Some(t1.min(t2))
    } else if t1 >= 0.0 {
        Some(t1)
    } else if t2 >= 0.0 {
        Some(t2)
    } else {
        None
    }
}

fn ray_segment_intersection(ray: &RayInner) -> Option<f64> {
    if ray.d.y.abs() <= 1e-12 {
        return None;
    }
    let t = -ray.o.y / ray.d.y;
    if t < 1e-12 {
        return None;
    }
    let x = ray.o.x + t * ray.d.x;
    if (-1.0..=1.0).contains(&x) {
        Some(t)
    } else {
        None
    }
}

fn ray_semicircle_intersection(ray: &RayInner) -> Option<f64> {
    let a = ray.d.dot(ray.d);
    let b = 2.0 * ray.o.dot(ray.d);
    let c = ray.o.dot(ray.o) - 1.0;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    let mut res: Option<f64> = None;
    for &t in [t1, t2].iter() {
        if t >= 1e-12 {
            let p = ray.offset(t);
            if p.y >= -1e-12 {
                res = match res {
                    Some(current) if current <= t => Some(current),
                    _ => Some(t),
                };
            }
        }
    }
    res
}

fn intersect_semicircle(ray: &RayInner, scale_y: f64) -> Option<f64> {
    if scale_y == 1.0 {
        return ray_semicircle_intersection(ray);
    }
    if scale_y <= 0.0 {
        return None;
    }
    let scaled_ray = RayInner {
        o: DVec2::new(ray.o.x, ray.o.y / scale_y),
        d: DVec2::new(ray.d.x, ray.d.y / scale_y),
    };
    ray_semicircle_intersection(&scaled_ray)
}

fn teleport_position(pos: DVec2, from_inv: &DMat3, to: &DMat3) -> DVec2 {
    let local = from_inv.transform_point2(pos);
    to.transform_point2(local)
}

fn teleport_direction(dir: DVec2, from_inv: &DMat3, to: &DMat3) -> DVec2 {
    let local = from_inv.transform_vector2(dir);
    to.transform_vector2(local)
}

fn circle_invert_ray_direction(ray: &RayInner) -> DVec2 {
    let p = ray.o;
    let d = ray.d;
    let r2 = p.dot(p);
    if r2 == 0.0 {
        return d;
    }
    let dot = p.dot(d);
    let num = d * r2 - p * (2.0 * dot);
    let denom = r2 * r2;
    let inv = num / denom;
    let orig_len = d.length();
    let inv_len = inv.length();
    if inv_len == 0.0 {
        DVec2::ZERO
    } else {
        inv * (orig_len / inv_len)
    }
}

fn ellipse_invert_ray_direction(ray: &RayInner, scale_y: f64) -> DVec2 {
    if scale_y <= 0.0 {
        return ray.d;
    }

    let p = ray.o;
    let nx = p.x;
    let ny = p.y / (scale_y * scale_y);

    let mut n = DVec2::new(nx, ny);
    let n_len = n.length();
    if n_len == 0.0 {
        return ray.d;
    }
    n /= n_len;

    let dot = ray.d.dot(n);
    ray.d - n * (2.0 * dot)
}

pub fn intersect_portal(
    ray: &RayInner,
    mut other_directions: Vec<DVec2>,
    portal: &Portal,
) -> Option<(bool, f64, RayInner, Vec<DVec2>)> {
    let local1 = ray.transform(&portal.portal1_inv);
    let local2 = ray.transform(&portal.portal2_inv);

    let (t1, t2) = match portal.kind {
        PortalType::Flat => (
            ray_segment_intersection(&local1),
            ray_segment_intersection(&local2),
        ),
        PortalType::Semicircle { scale_y } => (
            intersect_semicircle(&local1, scale_y),
            intersect_semicircle(&local2, scale_y),
        ),
        PortalType::Circle { scale_y }
        | PortalType::Perspective { scale_y }
        | PortalType::Wormhole { scale_y } => (
            if scale_y == 1.0 {
                ray_circle_intersection(&local1)
            } else {
                intersect_ellipse(&local1, scale_y)
            },
            if scale_y == 1.0 {
                ray_circle_intersection(&local2)
            } else {
                intersect_ellipse(&local2, scale_y)
            },
        ),
    };

    let (first, t_hit, from_inv, to, inv_to) = match (t1, t2) {
        (Some(t1), Some(t2)) => {
            if t1 < t2 {
                (
                    true,
                    t1,
                    &portal.portal1_inv,
                    &portal.portal2,
                    &portal.portal2_inv,
                )
            } else {
                (
                    false,
                    t2,
                    &portal.portal2_inv,
                    &portal.portal1,
                    &portal.portal1_inv,
                )
            }
        }
        (Some(t1), None) => (
            true,
            t1,
            &portal.portal1_inv,
            &portal.portal2,
            &portal.portal2_inv,
        ),
        (None, Some(t2)) => (
            false,
            t2,
            &portal.portal2_inv,
            &portal.portal1,
            &portal.portal1_inv,
        ),
        (None, None) => return None,
    };

    let new_pos = teleport_position(ray.offset(t_hit), from_inv, to);
    let mut new_dir = teleport_direction(ray.d, from_inv, to);
    other_directions
        .iter_mut()
        .for_each(|x| *x = teleport_direction(*x, from_inv, to));

    match portal.kind {
        PortalType::Wormhole { scale_y } => {
            let apply = |pos, dir| {
                let local_ray = RayInner::new(pos, dir).transform(inv_to);
                let inverted = if scale_y == 1.0 {
                    circle_invert_ray_direction(&local_ray)
                } else {
                    ellipse_invert_ray_direction(&local_ray, scale_y)
                };
                to.transform_vector2(inverted)
            };
            new_dir = apply(new_pos, new_dir);
            other_directions
                .iter_mut()
                .for_each(|x| *x = apply(new_pos, *x));
        }
        PortalType::Perspective { .. } => {
            new_dir = -new_dir;
            other_directions.iter_mut().for_each(|x| *x = -*x);
        }
        PortalType::Circle { .. } | PortalType::Semicircle { .. } | PortalType::Flat => {}
    }

    Some((
        first,
        t_hit,
        RayInner::new(new_pos, new_dir),
        other_directions,
    ))
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Particle {
    pub position: DVec2,
    pub velocity: DVec2,
    pub degree: [i8; 5], // max 5 portals
}

impl Particle {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            position: DVec2::new(x, y),
            velocity: DVec2::new(0.0, 0.0),
            degree: [0; 5],
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeSpring {
    pub i: usize,
    pub j: usize,
    pub rest_length: f64,
    pub died: bool,
    pub show: bool,
}

impl EdgeSpring {
    pub fn new(i: usize, j: usize, rest_length: f64, show: bool) -> Self {
        Self {
            i,
            j,
            rest_length,
            died: false,
            show,
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

fn reflect_around_unit_circle(pos: DVec2) -> DVec2 {
    let normal = pos.normalize();
    let len = pos.length();
    if len == 0. {
        return pos;
    }
    normal * 1. / len
}

fn reflect_direction_around_unit_circle(pos: DVec2, dir: DVec2) -> DVec2 {
    let r2 = pos.length_squared();
    if r2 == 0. {
        return dir;
    }
    // debug_assert!(r2 > 0.0, "Direction reflection undefined at the origin");

    let numerator = dir * r2 - pos * (2.0 * pos.dot(dir));
    let reflected = numerator / (r2 * r2);

    reflected.normalize() * dir.length()
}

impl Portal {
    fn new(a: DMat3, b: DMat3, portal_type: u8) -> Portal {
        Portal {
            kind: match portal_type {
                0 => PortalType::Wormhole { scale_y: 1.0 },
                1 => PortalType::Perspective { scale_y: 1.0 },
                2 => PortalType::Circle { scale_y: 1.0 },
                3 => PortalType::Semicircle { scale_y: 1.0 },
                4 => PortalType::Flat,
                _ => unreachable!(),
            },
            portal1: a,
            portal2: b,
            worlds: (0, 0),

            portal1_inv: a.inverse(),
            portal2_inv: b.inverse(),
        }
    }

    fn get_center1(&self) -> DVec2 {
        (self.portal1 * DVec3::new(0., 0., 1.)).xy()
    }

    fn get_center2(&self) -> DVec2 {
        (self.portal2 * DVec3::new(0., 0., 1.)).xy()
    }

    fn get_radius1(&self) -> f64 {
        (self.portal1 * DVec3::new(1., 0., 0.)).length()
    }

    fn get_radius2(&self) -> f64 {
        (self.portal2 * DVec3::new(1., 0., 0.)).length()
    }
}

fn teleport_position_full(portal: &Portal, pos: DVec2, mut degree: i8) -> DVec2 {
    let mut pos = DVec3::from((pos, 1.));
    loop {
        if degree == 0 {
            break;
        } else {
            if degree > 0 {
                pos = portal.portal1_inv * pos;
            } else {
                pos = portal.portal2_inv * pos;
            }

            match portal.kind {
                PortalType::Wormhole { scale_y } | PortalType::Perspective { scale_y } => {
                    if scale_y == 1.0 {
                        pos = DVec3::from((reflect_around_unit_circle(pos.xy()), 1.));
                    } else {
                        panic!("Ellipse is not supported");
                    }
                }
                PortalType::Circle { .. } | PortalType::Semicircle { .. } | PortalType::Flat => {}
            }

            if degree > 0 {
                degree -= 1;
                pos = portal.portal2 * pos;
            } else {
                degree += 1;
                pos = portal.portal1 * pos;
            }
        }
    }
    pos.xy()
}

fn teleport_direction_full(portal: &Portal, pos: DVec2, dir: DVec2, mut degree: i8) -> DVec2 {
    let mut pos = DVec3::from((pos, 1.));
    let mut dir = DVec3::from((dir, 0.));
    loop {
        if degree == 0 {
            break;
        } else {
            if degree > 0 {
                pos = portal.portal1_inv * pos;
                dir = portal.portal1_inv * dir;
            } else {
                pos = portal.portal2_inv * pos;
                dir = portal.portal2_inv * dir;
            }

            match portal.kind {
                PortalType::Wormhole { scale_y } => {
                    if scale_y == 1.0 {
                        pos = DVec3::from((reflect_around_unit_circle(pos.xy()), 1.));
                        dir = DVec3::from((
                            reflect_direction_around_unit_circle(pos.xy(), dir.xy()),
                            0.,
                        ));
                    } else {
                        panic!("Ellipse is not supported");
                    }
                }
                PortalType::Perspective { scale_y } => {
                    if scale_y == 1.0 {
                        dir = -dir;
                    } else {
                        panic!("Ellipse is not supported");
                    }
                }
                PortalType::Circle { .. } | PortalType::Semicircle { .. } | PortalType::Flat => {}
            }

            if degree > 0 {
                degree -= 1;
                pos = portal.portal2 * pos;
                dir = portal.portal2 * dir;
            } else {
                degree += 1;
                pos = portal.portal1 * pos;
                dir = portal.portal1 * dir;
            }
        }
    }
    dir.xy()
}

fn move_particle(portals: &[Portal], particle: &mut Particle, offset: DVec2) {
    let ray = RayInner::new(particle.position, offset);

    let res = portals
        .iter()
        .enumerate()
        .flat_map(|(i, portal)| Some((i, intersect_portal(&ray, vec![particle.velocity], portal)?)))
        .filter(|(_, (_, t, _, _))| *t <= 1.0 && *t >= 0.)
        .min_by_key(|(_, (_, t, _, _))| OrderedFloat(*t));

    if let Some((i, (dir, _, new_ray, velocity))) = res {
        particle.position = new_ray.o + new_ray.d;
        particle.velocity = velocity[0];
        if dir {
            particle.degree[i] += 1;
        } else {
            particle.degree[i] -= 1;
        }
    } else {
        particle.position += offset;
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

fn spring_force(
    pos1: DVec2,
    pos2: DVec2,
    speed1: DVec2,
    speed2: DVec2,
    rest_length: f64,
    edge_k: f64,
    damping: f64,
) -> DVec2 {
    let delta = pos2 - pos1;
    let mut current_length = delta.length();

    if current_length > rest_length * 6. {
        current_length = rest_length * 6.;
    }

    if current_length < rest_length * 0.1 {
        current_length = rest_length * 0.1;
    }

    let direction = delta * (1.0 / current_length);

    let relative_velocity = (speed2 - speed1).dot(direction);

    let spring_force = edge_k * (current_length - rest_length);
    let damping_force = damping * relative_velocity;
    let total_force = spring_force + damping_force;

    direction * total_force
}

fn calc_forces(
    particles: &[Particle],
    mut add_force: impl FnMut(usize, DVec2),
    springs: &Vec<EdgeSpring>,
    portals: &[Portal],
    edge_k: f64,
    damping: f64,
) {
    for spring in springs {
        if spring.died {
            continue;
        }

        let p1 = &particles[spring.i];
        let p2 = &particles[spring.j];

        if p1.degree != p2.degree {
            let idx = p1
                .degree
                .iter()
                .zip(p2.degree.iter())
                .enumerate()
                .find(|(_, (d1, d2))| d1 != d2)
                .unwrap()
                .0;
            let portal = &portals[idx];

            // in p1 local coordinates
            let force1 = spring_force(
                p1.position,
                teleport_position_full(portal, p2.position, p1.degree[idx] - p2.degree[idx]),
                p1.velocity,
                teleport_direction_full(
                    portal,
                    p2.position,
                    p2.velocity,
                    p1.degree[idx] - p2.degree[idx],
                ),
                spring.rest_length,
                edge_k,
                damping,
            );

            // in p2 local coordinates
            let force2 = spring_force(
                teleport_position_full(portal, p1.position, p2.degree[idx] - p1.degree[idx]),
                p2.position,
                teleport_direction_full(
                    portal,
                    p1.position,
                    p1.velocity,
                    p2.degree[idx] - p1.degree[idx],
                ),
                p2.velocity,
                spring.rest_length,
                edge_k,
                damping,
            );

            let force1_avg = (force1
                + teleport_direction_full(
                    portal,
                    p2.position,
                    force2,
                    p1.degree[idx] - p2.degree[idx],
                ))
                / 2.;
            let force2_avg = (force2
                + teleport_direction_full(
                    portal,
                    p1.position,
                    force1,
                    p2.degree[idx] - p1.degree[idx],
                ))
                / 2.;

            add_force(spring.i, force1_avg);
            add_force(spring.j, -force2_avg);
        } else {
            let force_vector = spring_force(
                p1.position,
                p2.position,
                p1.velocity,
                p2.velocity,
                spring.rest_length,
                edge_k,
                damping,
            );

            add_force(spring.i, force_vector);
            add_force(spring.j, -force_vector);
        };
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct PhysicsSystem {
    springs: Vec<EdgeSpring>,
    portals: Vec<Portal>,

    // Simulation constants
    dt: f64,
    edge_spring_constant: f64,
    damping_coefficient: f64,
    global_damping: f64,

    size: usize,
}

impl Default for PhysicsSystem {
    fn default() -> Self {
        PhysicsSystem {
            springs: Vec::new(),
            portals: vec![Portal::new(DMat3::IDENTITY, DMat3::IDENTITY, 0)],

            dt: 0.01,
            edge_spring_constant: 50.0,
            damping_coefficient: 10.,
            global_damping: 0.01,

            size: 30,
        }
    }
}

impl PhysicsSystem {
    fn init(
        &mut self,
        portals: Vec<Portal>,
        size: usize,
        scale: f64,
        offset: DVec2,
        speed: DVec2,
    ) -> Vec<Particle> {
        // Clear existing data
        self.springs.clear();
        self.portals = portals;

        let mut particles = vec![];

        for i in 0..self.size {
            for j in 0..self.size {
                let x = i as f64 / (self.size - 1) as f64;
                let y = j as f64 / (self.size - 1) as f64;
                let mut p = Particle::new(
                    x * scale - 0.5 * scale + offset.x,
                    y * scale - 0.5 * scale + offset.y,
                );
                p.velocity = DVec2::new(speed.x, speed.y);
                particles.push(p);
            }
        }

        self.size = size;

        let get_index = |i, j| i * self.size + j;
        let regular_len = 1. / (self.size - 1) as f64 * scale;
        let diagonal_len = regular_len * 2.0_f64.sqrt();
        let diag2_len = regular_len * 5.0_f64.sqrt();

        for i in 0..self.size {
            for j in 0..self.size {
                if i + 1 != self.size {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 1, j),
                        regular_len,
                        true,
                    ));
                }
                if j + 1 != self.size {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i, j + 1),
                        regular_len,
                        true,
                    ));
                }
                if i + 1 != self.size && j + 1 != self.size {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 1, j + 1),
                        diagonal_len,
                        false,
                    ));

                    self.springs.push(EdgeSpring::new(
                        get_index(i + 1, j),
                        get_index(i, j + 1),
                        diagonal_len,
                        false,
                    ));
                }

                if i + 2 < self.size && j + 1 < self.size {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 2, j + 1),
                        diag2_len,
                        false,
                    ));
                }

                if i + 1 < self.size && j + 2 < self.size {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 1, j + 2),
                        diag2_len,
                        false,
                    ));
                }

                if i + 2 < self.size && j > 0 {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 2, j - 1),
                        diag2_len,
                        false,
                    ));
                }

                if i + 1 < self.size && j > 1 {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i + 1, j - 2),
                        diag2_len,
                        false,
                    ));
                }
            }
        }

        particles
    }
}

// =============== Generic RK4 ===============

pub trait ODESystem<State, Deriv> {
    fn eval(&self, t: f64, state: &State, deriv_out: &mut Deriv);
}

pub trait RKOps<State, Deriv> {
    fn copy_state(&self, dst: &mut State, src: &State);
    fn build_from(&self, out: &mut State, base: &State, scale: f64, deriv: &Deriv);
    fn zero_deriv(&self, d: &mut Deriv);
    fn axpy_deriv(&self, out: &mut Deriv, a: f64, k: &Deriv);
    fn apply_final(&self, state: &mut State, combined: &Deriv, dt: f64);
}

#[derive(Clone, Debug)]
pub struct RK4Workspace<State, Deriv> {
    pub y0: State,
    pub k1: Deriv,
    pub k2: Deriv,
    pub k3: Deriv,
    pub k4: Deriv,
    pub sum: Deriv,
}

pub fn rk4_step<State, Deriv, System, Ops>(
    sys: &System,
    ops: &Ops,
    state: &mut State,
    t: f64,
    dt: f64,
    ws: &mut RK4Workspace<State, Deriv>,
) where
    System: ODESystem<State, Deriv>,
    Ops: RKOps<State, Deriv>,
{
    // snapshot
    ops.copy_state(&mut ws.y0, state);

    // k1
    sys.eval(t, state, &mut ws.k1);

    // k2
    ops.build_from(state, &ws.y0, 0.5 * dt, &ws.k1);
    sys.eval(t + 0.5 * dt, state, &mut ws.k2);

    // k3
    ops.build_from(state, &ws.y0, 0.5 * dt, &ws.k2);
    sys.eval(t + 0.5 * dt, state, &mut ws.k3);

    // k4
    ops.build_from(state, &ws.y0, dt, &ws.k3);
    sys.eval(t + dt, state, &mut ws.k4);

    // combine
    ops.zero_deriv(&mut ws.sum);
    ops.axpy_deriv(&mut ws.sum, 1.0, &ws.k1);
    ops.axpy_deriv(&mut ws.sum, 2.0, &ws.k2);
    ops.axpy_deriv(&mut ws.sum, 2.0, &ws.k3);
    ops.axpy_deriv(&mut ws.sum, 1.0, &ws.k4);

    // restore and apply final (portals happen here)
    ops.copy_state(state, &ws.y0);
    ops.apply_final(state, &ws.sum, dt);
}

// =============== Concrete wiring for your mesh ===============

#[derive(Clone, Copy, Debug, Default)]
pub struct Deriv2 {
    pub dp: DVec2, // d(position)/dt = velocity
    pub dv: DVec2, // d(velocity)/dt = acceleration (force / m) with m=1
}

impl Mesh {
    fn ensure_workspace(&mut self) {
        let n = self.particles.len();
        self.rk_ws.ensure_len(n);
    }

    pub fn integrate_rk4_2(&mut self, t: f64) {
        self.ensure_workspace();

        let ops = ParticleOps {
            portals: &self.system.portals,
        };

        rk4_step(
            &self.system,
            &ops,
            &mut self.particles,
            t,
            self.system.dt,
            &mut self.rk_ws,
        );

        // Post-step: spring death rule
        let spring_die_factor = 4.0;
        for spring in &mut self.system.springs {
            if !spring.died {
                let p1 = &self.particles[spring.i];
                let p2 = &self.particles[spring.j];
                if p1.degree == p2.degree
                    && (p1.position - p2.position).length() > spring.rest_length * spring_die_factor
                {
                    spring.died = true;
                }
            }
        }
    }
}

// ODESystem: how to compute derivatives for the mesh
impl ODESystem<Vec<Particle>, Vec<Deriv2>> for PhysicsSystem {
    fn eval(&self, _t: f64, state: &Vec<Particle>, deriv_out: &mut Vec<Deriv2>) {
        // forces buffer
        for f in &mut *deriv_out {
            f.dv = DVec2::ZERO;
        }

        let edge_k = self.edge_spring_constant * (self.size as f64) * (self.size as f64) / 100.0;

        calc_forces(
            state,
            |pos, force| deriv_out[pos].dv += force,
            &self.springs,
            &self.portals,
            edge_k,
            self.damping_coefficient,
        );

        // turn forces into derivatives
        if deriv_out.len() != state.len() {
            deriv_out.resize(state.len(), Deriv2::default());
        }

        for (i, p) in state.iter().enumerate() {
            let mut damped_force = deriv_out[i].dv - (p.velocity * self.global_damping);

            // clamp
            if damped_force.length() > 1e6 {
                damped_force = damped_force.normalize() * 1e6;
            }

            deriv_out[i] = Deriv2 {
                dp: p.velocity,
                dv: damped_force, // m = 1
            };
        }
    }
}

// RKOps for Vec<Particle> / Vec<Deriv2>
pub struct ParticleOps<'a> {
    pub portals: &'a [Portal],
}

impl<'a> RKOps<Vec<Particle>, Vec<Deriv2>> for ParticleOps<'a> {
    fn copy_state(&self, dst: &mut Vec<Particle>, src: &Vec<Particle>) {
        if dst.len() != src.len() {
            dst.resize(src.len(), Particle::default());
        }
        // element-wise copy (no alloc)
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.clone();
        }
    }

    fn build_from(
        &self,
        out: &mut Vec<Particle>,
        base: &Vec<Particle>,
        scale: f64,
        deriv: &Vec<Deriv2>,
    ) {
        debug_assert_eq!(out.len(), base.len());
        debug_assert_eq!(base.len(), deriv.len());
        for ((o, b), k) in out.iter_mut().zip(base.iter()).zip(deriv.iter()) {
            *o = b.clone(); // copy non-integrated fields (e.g., degree)
            o.velocity = b.velocity + k.dv * scale;
            o.position = b.position + k.dp * scale;
        }
    }

    fn zero_deriv(&self, d: &mut Vec<Deriv2>) {
        for di in d.iter_mut() {
            *di = Deriv2::default();
        }
    }

    fn axpy_deriv(&self, out: &mut Vec<Deriv2>, a: f64, k: &Vec<Deriv2>) {
        if out.len() != k.len() {
            out.resize(k.len(), Deriv2::default());
        }
        for (o, ki) in out.iter_mut().zip(k.iter()) {
            o.dp += ki.dp * a;
            o.dv += ki.dv * a;
        }
    }

    fn apply_final(&self, state: &mut Vec<Particle>, combined: &Vec<Deriv2>, dt: f64) {
        debug_assert_eq!(state.len(), combined.len());
        let s = dt / 6.0;
        for (p, k) in state.iter_mut().zip(combined.iter()) {
            p.velocity += k.dv * s;
            let offset = k.dp * s;
            move_particle(self.portals, p, offset); // portals only here
        }
    }
}

// Convenience: workspace alloc/resize specialized for particles
impl RK4Workspace<Vec<Particle>, Vec<Deriv2>> {
    pub fn new_particles(n: usize) -> Self {
        Self {
            y0: vec![Particle::default(); n],
            k1: vec![Deriv2::default(); n],
            k2: vec![Deriv2::default(); n],
            k3: vec![Deriv2::default(); n],
            k4: vec![Deriv2::default(); n],
            sum: vec![Deriv2::default(); n],
        }
    }

    pub fn ensure_len(&mut self, n: usize) {
        if self.y0.len() != n {
            self.y0.resize(n, Particle::default());
        }
        if self.k1.len() != n {
            self.k1.resize(n, Deriv2::default());
            self.k2.resize(n, Deriv2::default());
            self.k3.resize(n, Deriv2::default());
            self.k4.resize(n, Deriv2::default());
            self.sum.resize(n, Deriv2::default());
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Mesh {
    particles: Vec<Particle>,
    system: PhysicsSystem,

    // preallocated working memory
    rk_ws: RK4Workspace<Vec<Particle>, Vec<Deriv2>>,

    // Scene constants
    size: usize,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
    speed_x: f64,
    speed_y: f64,
    portal_type: u8,
    draw_reflections: bool,
    portal1_x: f64,
    portal1_y: f64,
    portal1_angle: f64,
    portal2_x: f64,
    portal2_y: f64,
    portal2_angle: f64,
    mirror_portals: bool,

    particles_buffer: Vec<f32>,
    lines_buffer: Vec<u32>,
    circle1data: Vec<f32>,
    circle2data: Vec<f32>,
    circle1data_teleported: Vec<f32>,
    circle2data_teleported: Vec<f32>,
    disable_lines_buffer: Vec<u8>,
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            system: Default::default(),

            rk_ws: RK4Workspace::<Vec<Particle>, Vec<Deriv2>>::new_particles(30 * 30),

            size: 30,
            scale: 1.,
            offset_x: -1.5,
            offset_y: 0.,
            speed_x: 0.5,
            speed_y: 0.,
            portal_type: 0,
            draw_reflections: true,
            portal1_x: -1.5,
            portal1_y: 0.,
            portal1_angle: 0.,
            portal2_x: 1.5,
            portal2_y: 0.,
            portal2_angle: 0.,
            mirror_portals: true,

            particles_buffer: Vec::new(),
            lines_buffer: Vec::new(),
            circle1data: Vec::new(),
            circle2data: Vec::new(),
            circle1data_teleported: Vec::new(),
            circle2data_teleported: Vec::new(),
            disable_lines_buffer: Vec::new(),
        }
    }

    pub fn init(&mut self) {
        self.particles = self.system.init(
            vec![Portal::new(
                DMat3::from_scale_angle_translation(
                    DVec2::new(1., 1.),
                    self.portal1_angle * PI,
                    DVec2::new(self.portal1_x, self.portal1_y),
                ),
                DMat3::from_scale_angle_translation(
                    if self.mirror_portals {
                        DVec2::new(-1., 1.)
                    } else {
                        DVec2::new(1., 1.)
                    },
                    self.portal2_angle * PI,
                    DVec2::new(self.portal2_x, self.portal2_y),
                ),
                self.portal_type,
            )],
            self.size,
            self.scale,
            DVec2::new(self.offset_x, self.offset_y),
            DVec2::new(self.speed_x, self.speed_y),
        );
        self.rk_ws =
            RK4Workspace::<Vec<Particle>, Vec<Deriv2>>::new_particles(self.particles.len());
        self.update_buffers();
    }

    pub fn step(&mut self) {
        self.update_buffers();
        self.integrate_rk4_2(0.);
    }

    pub fn get_constant(&self, name: &str) -> f32 {
        match name {
            "dt" => self.system.dt as f32,
            "edgeSpringConstant" => self.system.edge_spring_constant as f32,
            "dampingCoefficient" => self.system.damping_coefficient as f32,
            "globalDamping" => self.system.global_damping as f32,
            "draw_reflections" => self.draw_reflections as u32 as f32,

            "scene_size" => self.size as f32,
            "scene_scale" => self.scale as f32,
            "scene_offset_x" => self.offset_x as f32,
            "scene_offset_y" => self.offset_y as f32,
            "scene_speed_x" => self.speed_x as f32,
            "scene_speed_y" => self.speed_y as f32,
            "scene_prtal_type" => self.portal_type as f32,

            "scene_portal1_x" => self.portal1_x as f32,
            "scene_portal1_y" => self.portal1_y as f32,
            "scene_portal1_angle" => self.portal1_angle as f32,
            "scene_portal2_x" => self.portal2_x as f32,
            "scene_portal2_y" => self.portal2_y as f32,
            "scene_portal2_angle" => self.portal2_angle as f32,
            "scene_mirror_portals" => self.mirror_portals as u32 as f32,

            _ => -100500.,
        }
    }

    pub fn set_constant(&mut self, name: &str, value: f32) {
        match name {
            "dt" => self.system.dt = value as f64,
            "edgeSpringConstant" => self.system.edge_spring_constant = value as f64,
            "dampingCoefficient" => self.system.damping_coefficient = value as f64,
            "globalDamping" => self.system.global_damping = value as f64,
            "draw_reflections" => {
                self.draw_reflections = value > 0.5;
                self.update_buffers();
            }

            "scene_size" => {
                self.size = value as usize;
                self.init();
            }
            "scene_scale" => {
                self.scale = value as f64;
                self.init();
            }
            "scene_offset_x" => {
                self.offset_x = value as f64;
                self.init();
            }
            "scene_offset_y" => {
                self.offset_y = value as f64;
                self.init();
            }
            "scene_speed_x" => {
                self.speed_x = value as f64;
                self.init();
            }
            "scene_speed_y" => {
                self.speed_y = value as f64;
                self.init();
            }
            "scene_portal_type" => {
                self.portal_type = value as u8;
                self.init();
            }

            "scene_portal1_x" => {
                self.portal1_x = value as f64;
                self.init();
            }
            "scene_portal1_y" => {
                self.portal1_y = value as f64;
                self.init();
            }
            "scene_portal1_angle" => {
                self.portal1_angle = value as f64;
                self.init();
            }
            "scene_portal2_x" => {
                self.portal2_x = value as f64;
                self.init();
            }
            "scene_portal2_y" => {
                self.portal2_y = value as f64;
                self.init();
            }
            "scene_portal2_angle" => {
                self.portal2_angle = value as f64;
                self.init();
            }
            "scene_mirror_portals" => {
                self.mirror_portals = value as u32 == 1;
                self.init();
            }

            _ => {}
        }
    }

    pub fn update_buffers(&mut self) {
        self.particles_buffer.clear();
        for p in &self.particles {
            self.particles_buffer.push(p.position.x as f32);
            self.particles_buffer.push(p.position.y as f32);
        }

        if self.draw_reflections {
            for p in &self.particles {
                let new_pos = teleport_position_full(&self.system.portals[0], p.position, -1);
                self.particles_buffer.push(new_pos.x as f32);
                self.particles_buffer.push(new_pos.y as f32);
            }

            for p in &self.particles {
                let new_pos = teleport_position_full(&self.system.portals[0], p.position, 1);
                self.particles_buffer.push(new_pos.x as f32);
                self.particles_buffer.push(new_pos.y as f32);
            }
        }

        self.lines_buffer.clear();
        for spring in &self.system.springs {
            self.lines_buffer.push(spring.i as u32);
            self.lines_buffer.push(spring.j as u32);
        }

        if self.draw_reflections {
            for spring in &self.system.springs {
                if self.particles[spring.i].degree[0] + 1 == self.particles[spring.j].degree[0] {
                    self.lines_buffer.push(spring.i as u32);
                    self.lines_buffer
                        .push(self.particles.len() as u32 + spring.j as u32);
                } else if self.particles[spring.i].degree[0]
                    == self.particles[spring.j].degree[0] + 1
                {
                    self.lines_buffer
                        .push(self.particles.len() as u32 + spring.i as u32);
                    self.lines_buffer.push(spring.j as u32);
                } else {
                    self.lines_buffer
                        .push(self.particles.len() as u32 + spring.i as u32);
                    self.lines_buffer
                        .push(self.particles.len() as u32 + spring.j as u32);
                }
            }

            for spring in &self.system.springs {
                if self.particles[spring.i].degree[0] + 1 == self.particles[spring.j].degree[0] {
                    self.lines_buffer
                        .push(self.particles.len() as u32 * 2 + spring.i as u32);
                    self.lines_buffer.push(spring.j as u32);
                } else if self.particles[spring.i].degree[0]
                    == self.particles[spring.j].degree[0] + 1
                {
                    self.lines_buffer.push(spring.i as u32);
                    self.lines_buffer
                        .push(self.particles.len() as u32 * 2 + spring.j as u32);
                } else {
                    self.lines_buffer
                        .push(self.particles.len() as u32 * 2 + spring.i as u32);
                    self.lines_buffer
                        .push(self.particles.len() as u32 * 2 + spring.j as u32);
                }
            }
        }

        self.disable_lines_buffer.clear();
        for spring in &self.system.springs {
            self.disable_lines_buffer.push(
                (self.particles[spring.i].degree != self.particles[spring.j].degree
                    || spring.died
                    || !spring.show) as u8,
            );
        }

        if self.draw_reflections {
            for spring in &self.system.springs {
                self.disable_lines_buffer.push(
                    (if self.particles[spring.i].degree[0] != self.particles[spring.j].degree[0] {
                        (self.particles[spring.i].degree[0] - self.particles[spring.j].degree[0])
                            .abs()
                            != 1
                    } else {
                        false
                    } || spring.died
                        || !spring.show) as u8
                        + 2,
                );
            }

            for spring in &self.system.springs {
                self.disable_lines_buffer.push(
                    (if self.particles[spring.i].degree[0] != self.particles[spring.j].degree[0] {
                        (self.particles[spring.i].degree[0] - self.particles[spring.j].degree[0])
                            .abs()
                            != 1
                    } else {
                        false
                    } || spring.died
                        || !spring.show) as u8
                        + 4,
                );
            }
        }

        self.circle1data.clear();
        self.circle1data
            .push(self.system.portals[0].get_center1().x as f32);
        self.circle1data
            .push(self.system.portals[0].get_center1().y as f32);
        self.circle1data
            .push(self.system.portals[0].get_radius1() as f32);

        self.circle2data.clear();
        self.circle2data
            .push(self.system.portals[0].get_center2().x as f32);
        self.circle2data
            .push(self.system.portals[0].get_center2().y as f32);
        self.circle2data
            .push(self.system.portals[0].get_radius2() as f32);

        self.circle1data_teleported.clear();
        let center1 = self.system.portals[0].get_center1();
        let radius1_teleported = teleport_position_full(
            &self.system.portals[0],
            center1 + center1.normalize() * self.system.portals[0].get_radius1(),
            -1,
        );
        let radius1_teleported2 = teleport_position_full(
            &self.system.portals[0],
            center1 - center1.normalize() * self.system.portals[0].get_radius1(),
            -1,
        );
        let center1_teleported = (radius1_teleported + radius1_teleported2) / 2.;
        self.circle1data_teleported
            .push(center1_teleported.x as f32);
        self.circle1data_teleported
            .push(center1_teleported.y as f32);
        self.circle1data_teleported
            .push((radius1_teleported - radius1_teleported2).length() as f32 / 2.);

        self.circle2data_teleported.clear();
        let center2 = self.system.portals[0].get_center2();
        let radius2_teleported = teleport_position_full(
            &self.system.portals[0],
            center2 + center2.normalize() * self.system.portals[0].get_radius2(),
            1,
        );
        let radius2_teleported2 = teleport_position_full(
            &self.system.portals[0],
            center2 - center2.normalize() * self.system.portals[0].get_radius2(),
            1,
        );
        let center2_teleported = (radius2_teleported + radius2_teleported2) / 2.;
        self.circle2data_teleported
            .push(center2_teleported.x as f32);
        self.circle2data_teleported
            .push(center2_teleported.y as f32);
        self.circle2data_teleported
            .push((radius2_teleported - radius2_teleported2).length() as f32 / 2.);
    }

    pub fn get_particles_buffer(&mut self) -> *const f32 {
        self.particles_buffer.as_ptr()
    }

    pub fn get_particles_count(&self) -> u32 {
        if self.draw_reflections {
            self.particles.len() as u32 * 3
        } else {
            self.particles.len() as u32
        }
    }

    pub fn get_lines_buffer(&mut self) -> *const u32 {
        self.lines_buffer.as_ptr()
    }

    pub fn get_disable_lines_buffer(&mut self) -> *const u8 {
        self.disable_lines_buffer.as_ptr()
    }

    pub fn get_lines_count(&mut self) -> u32 {
        if self.draw_reflections {
            self.system.springs.len() as u32 * 3
        } else {
            self.system.springs.len() as u32
        }
    }

    pub fn get_circle1_data(&mut self) -> *const f32 {
        self.circle1data.as_ptr()
    }

    pub fn get_circle2_data(&mut self) -> *const f32 {
        self.circle2data.as_ptr()
    }

    pub fn get_circle1_teleported_data(&mut self) -> *const f32 {
        self.circle1data_teleported.as_ptr()
    }

    pub fn get_circle2_teleported_data(&mut self) -> *const f32 {
        self.circle2data_teleported.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct MeshHandle {
    mesh: Mesh,
}

impl Default for MeshHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl MeshHandle {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut result = Self { mesh: Mesh::new() };
        result.init();
        result
    }

    pub fn init(&mut self) {
        self.mesh.init();
        self.mesh.step();
        self.mesh.get_particles_buffer();
        self.mesh.get_lines_buffer();
        self.mesh.get_circle1_data();
        self.mesh.get_circle2_data();
    }

    pub fn step(&mut self) {
        self.mesh.step();
    }

    pub fn get_constant(&mut self, name: &str) -> f32 {
        self.mesh.get_constant(name)
    }

    pub fn set_constant(&mut self, name: &str, value: f32) {
        self.mesh.set_constant(name, value);
    }

    // Returns coordinates of particles, 2 floats per particle
    pub fn get_particles_buffer(&mut self) -> *const f32 {
        self.mesh.get_particles_buffer()
    }

    pub fn get_particles_count(&self) -> u32 {
        self.mesh.get_particles_count()
    }

    // Indexes of lines between particles, two numbers per line
    pub fn get_lines_buffer(&mut self) -> *const u32 {
        self.mesh.get_lines_buffer()
    }

    pub fn get_lines_count(&mut self) -> u32 {
        self.mesh.get_lines_count()
    }

    pub fn get_disable_lines_buffer(&mut self) -> *const u8 {
        self.mesh.get_disable_lines_buffer()
    }

    // Pointer to circle data in format: [circle1.center.x, circle1.center.y, circle1.radius]
    pub fn get_circle1_data(&mut self) -> *const f32 {
        self.mesh.get_circle1_data()
    }

    // Same, but for other circle
    pub fn get_circle2_data(&mut self) -> *const f32 {
        self.mesh.get_circle2_data()
    }

    pub fn get_circle1_teleported_data(&mut self) -> *const f32 {
        self.mesh.get_circle1_teleported_data()
    }

    // Same, but for other circle
    pub fn get_circle2_teleported_data(&mut self) -> *const f32 {
        self.mesh.get_circle2_teleported_data()
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn test1() {
        color_backtrace::install();

        let mut mesh = Mesh::new();
        mesh.init();

        mesh.size = 120;
        mesh.offset_x = -1.5;
        mesh.system.edge_spring_constant = 200.;

        // mesh.size = 120;
        // mesh.scale = 3.27;
        // mesh.offset_x = -4.49;
        // mesh.speed_x = 2.12;

        loop {
            mesh.step();
        }
    }
}

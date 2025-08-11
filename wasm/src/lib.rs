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

#[derive(Clone, Debug)]
pub struct Particle {
    pub position: DVec2,
    pub velocity: DVec2,
    pub force: DVec2,
    pub degree: [i8; 5], // max 5 portals
}

impl Particle {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            position: DVec2::new(x, y),
            velocity: DVec2::new(0.0, 0.0),
            force: DVec2::new(0.0, 0.0),
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
    particles: &mut [Particle],
    springs: &Vec<EdgeSpring>,
    portals: &[Portal],
    edge_k: f64,
    damping: f64,
) {
    for particle in particles.iter_mut() {
        particle.force = DVec2::new(0.0, 0.0);
    }

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

            particles[spring.i].force += force1_avg;
            particles[spring.j].force -= force2_avg;
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

            particles[spring.i].force += force_vector;
            particles[spring.j].force -= force_vector;
        };
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Mesh {
    particles: Vec<Particle>,
    springs: Vec<EdgeSpring>,
    portals: Vec<Portal>,

    // Simulation constants
    dt: f64,
    edge_spring_constant: f64,
    damping_coefficient: f64,
    global_damping: f64,

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
            springs: Vec::new(),
            portals: vec![Portal::new(DMat3::IDENTITY, DMat3::IDENTITY, 0)],

            dt: 0.01,
            edge_spring_constant: 50.0,
            damping_coefficient: 10.,
            global_damping: 0.01,

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
        // Clear existing data
        self.particles.clear();
        self.springs.clear();
        self.portals = vec![Portal::new(
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
        )];

        for i in 0..self.size {
            for j in 0..self.size {
                let x = i as f64 / (self.size - 1) as f64;
                let y = j as f64 / (self.size - 1) as f64;
                let mut p = Particle::new(
                    x * self.scale - 0.5 * self.scale + self.offset_x,
                    y * self.scale - 0.5 * self.scale + self.offset_y,
                );
                p.velocity = DVec2::new(self.speed_x, self.speed_y);
                self.particles.push(p);
            }
        }

        let get_index = |i, j| i * self.size + j;
        let regular_len = 1. / (self.size - 1) as f64 * self.scale;
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

        self.update_buffers();
    }

    pub fn step(&mut self) {
        self.update_buffers();
        self.integrate_rk4();
    }

    fn integrate_rk4(&mut self) {
        let original_states: Vec<Particle> = self.particles.to_vec();

        let mut k1 = vec![(DVec2::ZERO, DVec2::ZERO); self.particles.len()];
        let mut k2 = vec![(DVec2::ZERO, DVec2::ZERO); self.particles.len()];
        let mut k3 = vec![(DVec2::ZERO, DVec2::ZERO); self.particles.len()];
        let mut k4 = vec![(DVec2::ZERO, DVec2::ZERO); self.particles.len()];

        // STEP 1: First evaluation (k1) at the current state
        self.evaluate_derivatives(&mut k1);

        // STEP 2: Second evaluation (k2) at t + dt/2 using k1
        self.apply_derivatives_half_step(&k1, &original_states);
        self.evaluate_derivatives(&mut k2);

        // STEP 3: Third evaluation (k3) at t + dt/2 using k2
        self.restore_original_state(&original_states);
        self.apply_derivatives_half_step(&k2, &original_states);
        self.evaluate_derivatives(&mut k3);

        // STEP 4: Fourth evaluation (k4) at t + dt using k3
        self.restore_original_state(&original_states);
        self.apply_derivatives_full_step(&k3, &original_states);
        self.evaluate_derivatives(&mut k4);

        self.restore_original_state(&original_states);

        // STEP 5: Combine the derivatives with RK4 weights
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];

            let position_change = k1[i].0 * 1.0 + k2[i].0 * 2.0 + k3[i].0 * 2.0 + k4[i].0 * 1.0;
            let velocity_change = k1[i].1 * 1.0 + k2[i].1 * 2.0 + k3[i].1 * 2.0 + k4[i].1 * 1.0;

            p.velocity += velocity_change * (self.dt / 6.0);
            move_particle(&self.portals, p, position_change * (self.dt / 6.0));
        }

        let spring_die_factor = 4.;
        for spring in &mut self.springs {
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

    fn evaluate_derivatives(&mut self, derivatives: &mut [(DVec2, DVec2)]) {
        calc_forces(
            &mut self.particles,
            &self.springs,
            &self.portals,
            self.edge_spring_constant * (self.size as f64) * (self.size as f64) / 100.,
            self.damping_coefficient,
        );

        for (i, p) in self.particles.iter().enumerate() {
            let mut damped_force = p.force - (p.velocity * self.global_damping);

            if damped_force.length() > 1e6 {
                damped_force = damped_force.normalize() * 1e6;
            }

            derivatives[i] = (p.velocity, damped_force);
        }
    }

    fn apply_derivatives_half_step(
        &mut self,
        derivatives: &[(DVec2, DVec2)],
        original_states: &[Particle],
    ) {
        let half_dt = self.dt * 0.5;

        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            let original = &original_states[i];

            p.velocity = original.velocity + derivatives[i].1 * half_dt;
            move_particle(&self.portals, p, derivatives[i].0 * half_dt);
            // p.position += derivatives[i].0 * half_dt;
        }
    }

    fn apply_derivatives_full_step(
        &mut self,
        derivatives: &[(DVec2, DVec2)],
        original_states: &[Particle],
    ) {
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            let original = &original_states[i];

            p.velocity = original.velocity + derivatives[i].1 * self.dt;
            move_particle(&self.portals, p, derivatives[i].0 * self.dt);
            // p.position += derivatives[i].0 * self.dt;
        }
    }

    fn restore_original_state(&mut self, original_states: &[Particle]) {
        #[allow(clippy::manual_memcpy)]
        for i in 0..self.particles.len() {
            self.particles[i] = original_states[i].clone();
        }
    }

    pub fn get_constant(&self, name: &str) -> f32 {
        match name {
            "dt" => self.dt as f32,
            "edgeSpringConstant" => self.edge_spring_constant as f32,
            "dampingCoefficient" => self.damping_coefficient as f32,
            "globalDamping" => self.global_damping as f32,
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
            "dt" => self.dt = value as f64,
            "edgeSpringConstant" => self.edge_spring_constant = value as f64,
            "dampingCoefficient" => self.damping_coefficient = value as f64,
            "globalDamping" => self.global_damping = value as f64,
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
                let new_pos = teleport_position_full(&self.portals[0], p.position, -1);
                self.particles_buffer.push(new_pos.x as f32);
                self.particles_buffer.push(new_pos.y as f32);
            }

            for p in &self.particles {
                let new_pos = teleport_position_full(&self.portals[0], p.position, 1);
                self.particles_buffer.push(new_pos.x as f32);
                self.particles_buffer.push(new_pos.y as f32);
            }
        }

        self.lines_buffer.clear();
        for spring in &self.springs {
            self.lines_buffer.push(spring.i as u32);
            self.lines_buffer.push(spring.j as u32);
        }

        if self.draw_reflections {
            for spring in &self.springs {
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

            for spring in &self.springs {
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
        for spring in &self.springs {
            self.disable_lines_buffer.push(
                (self.particles[spring.i].degree != self.particles[spring.j].degree
                    || spring.died
                    || !spring.show) as u8,
            );
        }

        if self.draw_reflections {
            for spring in &self.springs {
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

            for spring in &self.springs {
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
            .push(self.portals[0].get_center1().x as f32);
        self.circle1data
            .push(self.portals[0].get_center1().y as f32);
        self.circle1data.push(self.portals[0].get_radius1() as f32);

        self.circle2data.clear();
        self.circle2data
            .push(self.portals[0].get_center2().x as f32);
        self.circle2data
            .push(self.portals[0].get_center2().y as f32);
        self.circle2data.push(self.portals[0].get_radius2() as f32);

        self.circle1data_teleported.clear();
        let center1 = self.portals[0].get_center1();
        let radius1_teleported = teleport_position_full(
            &self.portals[0],
            center1 + center1.normalize() * self.portals[0].get_radius1(),
            -1,
        );
        let radius1_teleported2 = teleport_position_full(
            &self.portals[0],
            center1 - center1.normalize() * self.portals[0].get_radius1(),
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
        let center2 = self.portals[0].get_center2();
        let radius2_teleported = teleport_position_full(
            &self.portals[0],
            center2 + center2.normalize() * self.portals[0].get_radius2(),
            1,
        );
        let radius2_teleported2 = teleport_position_full(
            &self.portals[0],
            center2 - center2.normalize() * self.portals[0].get_radius2(),
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
            self.springs.len() as u32 * 3
        } else {
            self.springs.len() as u32
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
        mesh.edge_spring_constant = 200.;

        // mesh.size = 120;
        // mesh.scale = 3.27;
        // mesh.offset_x = -4.49;
        // mesh.speed_x = 2.12;

        loop {
            mesh.step();
        }
    }
}

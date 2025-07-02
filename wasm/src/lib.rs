use glam::Vec3Swizzles;
use wasm_bindgen::prelude::*;

#[allow(non_camel_case_types)]
pub type fxx = f64;
pub type Vector2 = glam::DVec2;
pub type Vector3 = glam::DVec3;
pub type Matrix3 = glam::DMat3;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vector2,
    pub velocity: Vector2,
    pub force: Vector2,
    pub teleported: bool,
}

impl Particle {
    pub fn new(x: fxx, y: fxx) -> Self {
        Self {
            position: Vector2::new(x, y),
            velocity: Vector2::new(0.0, 0.0),
            force: Vector2::new(0.0, 0.0),
            teleported: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeSpring {
    pub i: usize,
    pub j: usize,
    pub rest_length: fxx,
    pub died: bool,
}

impl EdgeSpring {
    pub fn new(i: usize, j: usize, rest_length: fxx) -> Self {
        Self { i, j, rest_length, died: false }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Portals {
    a: Matrix3,
    b: Matrix3,

    a_inv: Matrix3,
    b_inv: Matrix3,
}

#[derive(Clone, Debug, Copy, PartialEq)]
enum TeleportDirection {
    A2B,
    B2A,
}

impl TeleportDirection {
    fn from_bool(val: bool) -> TeleportDirection {
        use TeleportDirection::*;
        match val {
            false => B2A,
            true => A2B,
        }
    }

    fn inv(self) -> Self {
        use TeleportDirection::*;
        match self {
            A2B => B2A,
            B2A => A2B,
        }
    }
}

fn is_intersects_unit_circle(a: Vector2, b: Vector2) -> bool {
    let a_dist_sq = a.length_squared();
    let b_dist_sq = b.length_squared();
    
    // Check if either endpoint is exactly on the circle
    if (a_dist_sq - 1.0).abs() < fxx::EPSILON || (b_dist_sq - 1.0).abs() < fxx::EPSILON {
        return true;
    }
    
    // If both points are inside the circle, no intersection with border
    if a_dist_sq < 1.0 && b_dist_sq < 1.0 {
        return false;
    }
    
    // If one point is inside and one outside, must cross the border
    if (a_dist_sq < 1.0) != (b_dist_sq < 1.0) {
        return true;
    }
    
    // Both points are outside the circle
    // Solve the quadratic equation for line-circle intersection
    let ab = b - a;
    let ab_len_sq = ab.length_squared();
    
    // Handle degenerate case (same point)
    if ab_len_sq < fxx::EPSILON {
        return false;
    }
    
    // Quadratic equation: |a + t(b-a)|² = 1
    // Expanded: |a|² + 2t⟨a,b-a⟩ + t²|b-a|² = 1
    // Rearranged: |b-a|²t² + 2⟨a,b-a⟩t + (|a|² - 1) = 0
    let c = a_dist_sq - 1.0;  // constant term
    let b_coeff = 2.0 * a.dot(ab);  // linear coefficient
    let a_coeff = ab_len_sq;  // quadratic coefficient
    
    let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c;
    
    // No real solutions = no intersection
    if discriminant < 0.0 {
        return false;
    }
    
    // Calculate the two solutions
    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b_coeff - sqrt_discriminant) / (2.0 * a_coeff);
    let t2 = (-b_coeff + sqrt_discriminant) / (2.0 * a_coeff);
    
    // Check if either intersection point lies on the segment [0, 1]
    (t1 >= 0.0 && t1 <= 1.0) || (t2 >= 0.0 && t2 <= 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_endpoint_inside_one_outside() {
        // One point inside, one outside - must cross border
        assert!(is_intersects_unit_circle(Vector2::new(0.5, 0.0), Vector2::new(2.0, 0.0)));
        assert!(is_intersects_unit_circle(Vector2::new(2.0, 0.0), Vector2::new(0.5, 0.0)));
    }

    #[test]
    fn test_both_endpoints_inside_circle() {
        // Both points inside circle - no border intersection
        assert!(!is_intersects_unit_circle(Vector2::new(0.5, 0.0), Vector2::new(0.0, 0.5)));
        assert!(!is_intersects_unit_circle(Vector2::new(-0.3, 0.3), Vector2::new(0.2, -0.4)));
    }

    #[test]
    fn test_endpoint_on_circle() {
        // Point on circle boundary
        assert!(is_intersects_unit_circle(Vector2::new(1.0, 0.0), Vector2::new(2.0, 0.0)));
        assert!(is_intersects_unit_circle(Vector2::new(0.0, 1.0), Vector2::new(0.0, 2.0)));
    }

    #[test]
    fn test_segment_tangent_to_circle() {
        // Segment tangent to circle (touches border at exactly one point)
        assert!(is_intersects_unit_circle(Vector2::new(-1.0, 1.0), Vector2::new(1.0, 1.0)));
        assert!(is_intersects_unit_circle(Vector2::new(1.0, -1.0), Vector2::new(1.0, 1.0)));
    }

    #[test]
    fn test_segment_crosses_circle_completely() {
        // Segment passes through circle (enters and exits)
        assert!(is_intersects_unit_circle(Vector2::new(-2.0, 0.0), Vector2::new(2.0, 0.0)));
        assert!(is_intersects_unit_circle(Vector2::new(0.0, -2.0), Vector2::new(0.0, 2.0)));
    }

    #[test]
    fn test_segment_misses_circle() {
        // Segment completely outside circle, doesn't touch border
        assert!(!is_intersects_unit_circle(Vector2::new(2.0, 2.0), Vector2::new(3.0, 2.0)));
        assert!(!is_intersects_unit_circle(Vector2::new(1.5, 0.0), Vector2::new(2.0, 0.0)));
    }
}

fn reflect_around_unit_circle(pos: Vector2) -> Vector2 {
    let normal = pos.normalize();
    let len = pos.length();
    normal * 1. / len
}

fn reflect_direction_around_unit_circle(pos: Vector2, dir: Vector2) -> Vector2 {
    let r2 = pos.length_squared();
    debug_assert!(r2 > 0.0, "Direction reflection undefined at the origin");

    let numerator = dir * r2 - pos * (2.0 * pos.dot(dir));
    let reflected = numerator / (r2 * r2);

    reflected.normalize() * dir.length()
}


impl Portals {
    fn new(a: Matrix3, b: Matrix3) -> Portals {
        Self {
            a,
            b,
            a_inv: a.inverse(),
            b_inv: b.inverse(),
        }
    }

    fn get_center1(&self) -> Vector2 {
        (self.a * Vector3::new(0., 0., 1.)).xy()
    }

    fn get_center2(&self) -> Vector2 {
        (self.b * Vector3::new(0., 0., 1.)).xy()
    }

    fn get_radius1(&self) -> fxx {
        (self.a * Vector3::new(1., 0., 0.)).length()
    }

    fn get_radius2(&self) -> fxx {
        (self.b * Vector3::new(1., 0., 0.)).length()
    }

    fn teleport_position(&self, pos: Vector2, teleport_direction: TeleportDirection) -> Vector2 {
        let mut vec = Vector3::from((pos, 1.));
        if teleport_direction == TeleportDirection::A2B {
            vec = self.a_inv * vec;
            vec = Vector3::from((reflect_around_unit_circle(vec.xy()), vec.z));
            vec = self.b * vec;
            vec.xy()
        } else {
            vec = self.b_inv * vec;
            vec = Vector3::from((reflect_around_unit_circle(vec.xy()), vec.z));
            vec = self.a * vec;
            vec.xy()
        }
    }

    fn teleport_direction(&self, pos: Vector2, dir: Vector2, teleport_direction: TeleportDirection) -> Vector2 {
        let mut pos = Vector3::from((pos, 1.));
        let mut dir = Vector3::from((dir, 0.));
        let res = if teleport_direction == TeleportDirection::A2B {
            pos = self.a_inv * pos;
            dir = self.a_inv * dir;
            dir = Vector3::from((reflect_direction_around_unit_circle(pos.xy(), dir.xy()), 0.));
            dir = self.b * dir;
            dir.xy()
        } else {
            pos = self.b_inv * pos;
            dir = self.b_inv * dir;
            dir = Vector3::from((reflect_direction_around_unit_circle(pos.xy(), dir.xy()), 0.));
            dir = self.a * dir;
            dir.xy()
        };
        res
    }

    fn is_intersect(&self, prev_pos: Vector2, new_pos: Vector2) -> Option<TeleportDirection> {
        use TeleportDirection::*;
        let prev_pos: Vector3 = (prev_pos, 1.0).into();
        let new_pos: Vector3 = (new_pos, 1.0).into();
        if is_intersects_unit_circle((self.a_inv * prev_pos).xy(), (self.a_inv * new_pos).xy()) {
            Some(A2B)
        } else if is_intersects_unit_circle((self.b_inv * prev_pos).xy(), (self.b_inv * new_pos).xy()) {
            Some(B2A)
        } else {
            None
        }
    }

    fn move_particle(&self, particle: &mut Particle, offset: Vector2) {
        if let Some(dir) = self.is_intersect(particle.position, particle.position + offset) {
            use TeleportDirection::*;
            particle.velocity = self.teleport_direction(particle.position + offset, particle.velocity, dir);
            particle.position = self.teleport_position(particle.position + offset, dir);
            if dir == A2B {
                particle.teleported = true;
            } else {
                particle.teleported = false;
            }
        } else {
            particle.position += offset;
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

fn spring_force(
    pos1: Vector2,
    pos2: Vector2,
    speed1: Vector2,
    speed2: Vector2,
    rest_length: fxx,
    edge_k: fxx,
    damping: fxx,
) -> Vector2 {
    let delta = pos2 - pos1;
    let current_length = delta.length();

    if current_length > 0.5 {
        // dbg!(pos1, pos2, speed1, speed2, rest_length, current_length);
        // panic!()
        return Vector2::ZERO;
    }

    if current_length == 0.0 {
        return Vector2::ZERO;
    }

    let direction = delta * (1.0 / current_length);

    let relative_velocity = (speed2 - speed1).dot(direction);

    let spring_force = edge_k * (current_length - rest_length);
    let damping_force = damping * relative_velocity;
    let total_force = spring_force + damping_force;

    let force_vector = direction * total_force;

    force_vector
}

fn calc_forces(
    particles: &mut Vec<Particle>,
    springs: &Vec<EdgeSpring>,
    portals: &Portals,
    edge_k: fxx,
    damping: fxx,
) {
    for particle in particles.iter_mut() {
        particle.force = Vector2::new(0.0, 0.0);
    }

    for spring in springs {
        if spring.died {
            continue;
        }

        let p1 = &particles[spring.i];
        let p2 = &particles[spring.j];

        if p1.teleported != p2.teleported {
            let force1 = spring_force(
                p1.position,
                portals.teleport_position(p2.position, TeleportDirection::from_bool(p2.teleported).inv()),
                p1.velocity,
                portals.teleport_direction(p2.position, p2.velocity, TeleportDirection::from_bool(p2.teleported).inv()),
                spring.rest_length,
                edge_k,
                damping
            );

            let force2 = spring_force(
                portals.teleport_position(p1.position, TeleportDirection::from_bool(p1.teleported).inv()),
                p2.position,
                portals.teleport_direction(p1.position, p1.velocity, TeleportDirection::from_bool(p1.teleported).inv()),
                p2.velocity,
                spring.rest_length,
                edge_k,
                damping
            );

            particles[spring.i].force += force1;
            particles[spring.j].force -= force2;
        } else {
            let force_vector = spring_force(p1.position, p2.position, p1.velocity, p2.velocity, spring.rest_length, edge_k, damping);

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
    portals: Portals,
    sizex: usize,
    sizey: usize,

    // Simulation constants
    dt: fxx,
    edge_spring_constant: fxx,
    damping_coefficient: fxx,
    global_damping: fxx,

    particles_buffer: Vec<f32>,
    lines_buffer: Vec<u32>,
    circle1data: Vec<f32>,
    circle2data: Vec<f32>,
    disable_lines_buffer: Vec<u8>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            springs: Vec::new(),
            portals: Portals::new(
                Matrix3::from_scale_angle_translation(
                    Vector2::new(1., 1.),
                    0.,
                    Vector2::new(-1.5, 0.),
                ),
                Matrix3::from_scale_angle_translation(
                    Vector2::new(-1., -1.),
                    0.,
                    Vector2::new(1.5, 0.),
                )
            ),
            sizex: 0,
            sizey: 0,

            dt: 0.01,
            edge_spring_constant: 50.0,
            damping_coefficient: 0.5,
            global_damping: 0.01,

            particles_buffer: Vec::new(),
            lines_buffer: Vec::new(),
            circle1data: Vec::new(),
            circle2data: Vec::new(),
            disable_lines_buffer: Vec::new(),
        }
    }

    pub fn init(&mut self, sizex: usize, sizey: usize, _scene: &str) {
        // Clear existing data
        self.particles.clear();
        self.springs.clear();
        self.sizex = sizex;
        self.sizey = sizey;

        let sizea = sizex.min(sizey);

        let scale = 1.;

        for i in 0..sizex {
            for j in 0..sizey {
                let x = i as fxx / (sizea - 1) as fxx;
                let y = j as fxx / (sizea - 1) as fxx;
                let mut p = Particle::new(x*scale - 4.7, y*scale - 0.5 * scale);
                p.velocity = Vector2::new(0.5, 0.);
                self.particles.push(p);
            }
        }

        let get_index = |i, j| i * sizey + j;
        let regular_len = 1. / (sizea - 1) as fxx * scale;
        let diagonal_len = regular_len * (2.0 as fxx).sqrt();
        let diag2_len = regular_len * (5.0 as fxx).sqrt();

        for i in 0..sizex {
            for j in 0..sizey {
                if i+1 != sizex {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+1, j),
                        regular_len
                    ));
                }
                if j+1 != sizey {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i, j+1),
                        regular_len
                    ));
                }
                if i+1 != sizex && j+1 != sizey {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+1, j+1),
                        diagonal_len
                    ));

                    self.springs.push(EdgeSpring::new(
                        get_index(i+1, j),
                        get_index(i, j+1),
                        diagonal_len
                    ));
                }

                if i+2 < sizex && j+1 < sizey {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+2, j+1),
                        diag2_len
                    ));
                }

                if i+1 < sizex && j+2 < sizey {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+1, j+2),
                        diag2_len
                    ));
                }

                if i+2 < sizex && j > 0 {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+2, j-1),
                        diag2_len
                    ));
                }

                if i+1 < sizex && j > 1 {
                    self.springs.push(EdgeSpring::new(
                        get_index(i, j),
                        get_index(i+1, j-2),
                        diag2_len
                    ));
                }
            }
        }
    }

    // Step the simulation forward in time
    pub fn step(&mut self) {
        self.integrate_rk4();
    }

    // Runge-Kutta 4 integration
    fn integrate_rk4(&mut self) {
        // Store original state
        let original_states: Vec<(Vector2, Vector2, bool)> = self
            .particles
            .iter()
            .map(|p| (p.position, p.velocity, p.teleported))
            .collect();

        // Arrays for the 4 evaluations
        let mut k1 = vec![(Vector2::ZERO, Vector2::ZERO); self.particles.len()];
        let mut k2 = vec![(Vector2::ZERO, Vector2::ZERO); self.particles.len()];
        let mut k3 = vec![(Vector2::ZERO, Vector2::ZERO); self.particles.len()];
        let mut k4 = vec![(Vector2::ZERO, Vector2::ZERO); self.particles.len()];

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

        // Restore to original state before final update
        self.restore_original_state(&original_states);

        // STEP 5: Combine the derivatives with RK4 weights
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];

            // Update position
            let position_change = k1[i].0 * 1.0 + k2[i].0 * 2.0 + k3[i].0 * 2.0 + k4[i].0 * 1.0;

            // Update velocity
            let velocity_change = k1[i].1 * 1.0 + k2[i].1 * 2.0 + k3[i].1 * 2.0 + k4[i].1 * 1.0;

            p.velocity = p.velocity + velocity_change * (self.dt / 6.0);
            self.portals.move_particle(p, position_change * (self.dt / 6.0));
        }

        let spring_die_factor = 3.;
        for spring in &mut self.springs {
            if !spring.died {
                let p1 = &self.particles[spring.i];
                let p2 = &self.particles[spring.j];

                if p1.teleported == p2.teleported && (p1.position - p2.position).length() > spring.rest_length * spring_die_factor {
                    spring.died = true;
                }
            }
        }
    }

    // Evaluate derivatives at current state
    fn evaluate_derivatives(&mut self, derivatives: &mut Vec<(Vector2, Vector2)>) {
        // Calculate forces
        calc_forces(
            &mut self.particles,
            &self.springs,
            &self.portals,
            self.edge_spring_constant,
            self.damping_coefficient,
        );

        // Store derivatives
        for (i, p) in self.particles.iter().enumerate() {
            // Apply global damping
            let damped_force = p.force - (p.velocity * self.global_damping);

            derivatives[i] = (p.velocity, damped_force);
        }
    }

    // Apply half-step changes
    fn apply_derivatives_half_step(
        &mut self,
        derivatives: &Vec<(Vector2, Vector2)>,
        original_states: &Vec<(Vector2, Vector2, bool)>,
    ) {
        let half_dt = self.dt * 0.5;

        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            let original = &original_states[i];

            p.velocity = original.1 + derivatives[i].1 * half_dt;
            self.portals.move_particle(p, derivatives[i].0 * half_dt);
        }
    }

    // Apply full-step changes
    fn apply_derivatives_full_step(
        &mut self,
        derivatives: &Vec<(Vector2, Vector2)>,
        original_states: &Vec<(Vector2, Vector2, bool)>,
    ) {
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            let original = &original_states[i];

            p.velocity = original.1 + derivatives[i].1 * self.dt;
            self.portals.move_particle(p, derivatives[i].0 * self.dt);
        }
    }

    // Restore original state
    fn restore_original_state(&mut self, original_states: &Vec<(Vector2, Vector2, bool)>) {
        for i in 0..self.particles.len() {
            let p = &mut self.particles[i];
            let original = &original_states[i];

            p.position = original.0;
            p.velocity = original.1;
            p.teleported = original.2;
        }
    }

    // Get constants
    pub fn get_constants(&self) -> (fxx, fxx, fxx, fxx) {
        (
            self.dt,
            self.edge_spring_constant,
            self.damping_coefficient,
            self.global_damping,
        )
    }

    // Set constant
    pub fn set_constant(&mut self, name: &str, value: f32) {
        match name {
            "dt" => self.dt = value as fxx,
            "edgeSpringConstant" => self.edge_spring_constant = value as fxx,
            "dampingCoefficient" => self.damping_coefficient = value as fxx,
            "globalDamping" => self.global_damping = value as fxx,
            _ => {}
        }
    }

    pub fn get_particles_buffer(&mut self) -> *const f32 {
        self.particles_buffer.clear();
        for p in &self.particles {
            self.particles_buffer.push(p.position.x as f32);
            self.particles_buffer.push(p.position.y as f32);
        }
        self.particles_buffer.as_ptr()
    }

    pub fn get_particles_count(&self) -> u32 {
        self.particles.len() as u32
    }

    pub fn get_lines_buffer(&mut self) -> *const u32 {
        self.lines_buffer.clear();
        for spring in &self.springs {
            self.lines_buffer.push(spring.i as u32);
            self.lines_buffer.push(spring.j as u32);
        }
        self.lines_buffer.as_ptr()
    }

    pub fn get_disable_lines_buffer(&mut self) -> *const u8 {
        self.disable_lines_buffer.clear();
        for spring in &self.springs {
            self.disable_lines_buffer.push((self.particles[spring.i].teleported != self.particles[spring.j].teleported || spring.died) as u8);
        }
        self.disable_lines_buffer.as_ptr()
    }

    pub fn get_lines_count(&mut self) -> u32 {
        self.lines_buffer.len() as u32
    }

    pub fn get_circle1_data(&mut self) -> *const f32 {
        self.circle1data.clear();
        self.circle1data.push(self.portals.get_center1().x as f32);
        self.circle1data.push(self.portals.get_center1().y as f32);
        self.circle1data.push(self.portals.get_radius1() as f32);
        self.circle1data.as_ptr()
    }

    pub fn get_circle2_data(&mut self) -> *const f32 {
        self.circle2data.clear();
        self.circle2data.push(self.portals.get_center2().x as f32);
        self.circle2data.push(self.portals.get_center2().y as f32);
        self.circle2data.push(self.portals.get_radius2() as f32);
        self.circle2data.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct MeshHandle {
    mesh: Mesh,
}

#[wasm_bindgen]
impl MeshHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(sizex: usize, sizey: usize, scene: &str) -> Self {
        let mut mesh = Mesh::new();
        mesh.init(sizex, sizey, scene);
        mesh.step();
        mesh.get_particles_buffer();
        mesh.get_lines_buffer();
        mesh.get_circle1_data();
        mesh.get_circle2_data();

        Self { mesh }
    }

    pub fn step(&mut self) {
        self.mesh.step();
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
}

#[cfg(test)]
mod tests2 {
    use super::*;

    #[test]
    fn test1() {
        color_backtrace::install();

        let mut mesh = Mesh::new();
        mesh.init(20, 20, "default");

        loop {
            mesh.step();
        }
    }
}
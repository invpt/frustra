// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use nalgebra::{Matrix4, Rotation3, Vector3};

use std::{collections::HashSet, sync::Arc, time::Instant};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Fullscreen, WindowBuilder},
};

pub(crate) mod math;
mod renderer;
pub(crate) mod world;

fn main() {
    let event_loop = EventLoop::new();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    window
        .set_cursor_grab(
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            CursorGrabMode::Locked,
            #[cfg(any(target_os = "windows"))]
            CursorGrabMode::Confined,
        )
        .unwrap();
    window.set_cursor_visible(false);

    /*let object = crate::world::Object::new(32, 32, 32, |x, y, z| {
        if y == 10 {
            return true;
        }
        let (x, y, z) = (x as f32, y as f32, z as f32);
        let (x, y, z) = (x - 16.0, y - 16.0, z - 16.0);
        (x * x + y * y + z * z) < 8.0f32.powi(2)
    });*/
    let mut object = crate::world::Object::new(48, 48, 48, |x, y, z| {
        if y == 0 {
            return true;
        }

        if y < 5 && (x == 0 || x == 11 || z == 0 || z == 11) {
            return true;
        }

        if y > 7 && (x == 0 || x == 11 || z == 0 || z == 11) {
            return true;
        }

        if (x == 0 || x == 11) && (z == 0 || z == 11) {
            return true;
        }

        y == 11 && x < 12 && z < 12
    });
    let blocks = [
        // floor
    ];
    for (x, y, z) in blocks {
        *object.get_mut(x, y, z).unwrap() = true;
    }
    /*let mut object = crate::world::Object::new(48, 48, 48, |_, _, _| false);
    renderer::dda::dda(Vector3::new(8.5, 0.5, 47.5), Vector3::new(-0.57735026,
        0.57735026,
        -0.57735026,), |x, y, z| {
            let (Ok(x), Ok(y), Ok(z)) = (x.try_into(), y.try_into(), z.try_into()) else { return renderer::dda::TraversalAction::Stop };
            if let Some(b) = object.get_mut(x, y, z) {
                *b = true;
                renderer::dda::TraversalAction::Continue
            } else {
                renderer::dda::TraversalAction::Stop
            }
        });*/
    let mut renderer = renderer::Renderer::new(
        &event_loop,
        window.clone(),
        window.inner_size().into(),
        &object,
    )
    .unwrap();

    let mut pressed = HashSet::<VirtualKeyCode>::new();

    let mut t = Instant::now();

    let sensitivity = 0.005;

    let mut cam_pos = (0.0, 5.5, 0.0);
    let mut rot_y = 0.0f32;
    let mut rot_x = 0.0f32;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                renderer.recreate_swapchain();
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if let Some(kc) = input.virtual_keycode {
                    if input.state == ElementState::Pressed {
                        if kc == VirtualKeyCode::Escape {
                            *control_flow = ControlFlow::Exit;
                        }
                        if kc == VirtualKeyCode::F11 {
                            if window.fullscreen().is_some() {
                                window.set_fullscreen(None)
                            } else {
                                window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                            }
                        }
                    }

                    if input.state == ElementState::Pressed {
                        pressed.insert(kc);
                    } else {
                        pressed.remove(&kc);
                    }
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                rot_y += sensitivity * delta.0 as f32;
                rot_x += sensitivity * delta.1 as f32;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                let look_direction = Rotation3::new(Vector3::new(0.0, -rot_y, 0.0))
                    * Rotation3::new(Vector3::new(-rot_x, 0.0, 0.0))
                    * Vector3::new(0.0, 0.0, -1.0);

                if let Some((x, y, z)) = object.cast_ray(
                    Vector3::new(cam_pos.0, cam_pos.1, cam_pos.2),
                    look_direction,
                ) {
                    *object.get_mut(x, y, z).unwrap() = false;
                    renderer.removed(&object, x, y, z);
                }
            }
            Event::RedrawEventsCleared => {
                let now = Instant::now();
                let dt = now.duration_since(t).as_secs_f32();
                t = now;

                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                let image_extent: [u32; 2] = window.inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }

                let mut move_dir = Vector3::new(0.0, 0.0, 0.0);
                if pressed.contains(&VirtualKeyCode::W) {
                    move_dir.z -= 1.0;
                }
                if pressed.contains(&VirtualKeyCode::S) {
                    move_dir.z += 1.0;
                }
                if pressed.contains(&VirtualKeyCode::A) {
                    move_dir.x -= 1.0;
                }
                if pressed.contains(&VirtualKeyCode::D) {
                    move_dir.x += 1.0;
                }
                if pressed.contains(&VirtualKeyCode::Q) {
                    move_dir.y -= 1.0;
                }
                if pressed.contains(&VirtualKeyCode::E) {
                    move_dir.y += 1.0;
                }
                if move_dir != Vector3::new(0.0, 0.0, 0.0) {
                    move_dir.normalize_mut();
                }

                move_dir = Rotation3::new(Vector3::new(0.0, -rot_y, 0.0)) * move_dir;

                let move_speed = 12.951 * dt;

                cam_pos.0 += move_speed * move_dir.x;
                cam_pos.1 += move_speed * move_dir.y;
                cam_pos.2 += move_speed * move_dir.z;

                renderer.draw(
                    image_extent,
                    Matrix4::new_rotation(Vector3::new(rot_x, 0.0, 0.0))
                        * Matrix4::new_rotation(Vector3::new(0.0, rot_y, 0.0))
                        * Matrix4::new_translation(&-Vector3::new(cam_pos.0, cam_pos.1, cam_pos.2)),
                );
            }
            _ => (),
        }
    });
}

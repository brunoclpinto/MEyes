//
//  Camera.controller.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

/// Abstract definition of content for the camera controllers by exposing expected public access.
/// Every other definition in the individual CameraControllers must be private.
public protocol CameraController: Actor {
  var cameras: [Camera] {get}
  
  init() async
}

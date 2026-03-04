//
//  Camera.controller.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

import Foundation

/// Abstract definition of content for the camera controllers by exposing expected public access.
/// Every other definition in the individual CameraControllers must be private.
public protocol CameraController: Actor {
  var cameras: [any Camera] {get}
  
  init() async
}

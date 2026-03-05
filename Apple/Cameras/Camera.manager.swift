//
//  Cameras.controller.swift
//  MEyes
//
//  Created by Bruno Pinto on 24/02/2026.
//

/// Serves as bridge between host and cameras, making it agnostic to camera requirements in terms of detection, connection and feed manipulation.

import Foundation

public actor CameraManager {
  public var controllers:[CameraController] = []
  
  public func load() async {
    controllers = []
    await withTaskGroup(of: CameraController.self) { group in
      for controllerType in CameraManager.knownControllers {
        group.addTask {
          await controllerType.init()
        }
      }
      
      for await controller in group {
        controllers.append(controller)
      }
    }
  }
}

extension CameraManager {
  /// List of CameraControllers one wishes to use.
  private static let knownControllers: [any CameraController.Type] = [
    CameraControllerIphone.self,
    CameraControllerMeta.self
  ]
}

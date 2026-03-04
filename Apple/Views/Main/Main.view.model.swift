//
//  Main.view.model.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import SwiftUI
internal import Combine

@MainActor
class MainViewModel: ObservableObject {
  @Published public private(set) var cameras: [CameraSnapshot] = []
  
  private let cameraManager = CameraManager()
  
  func load() async {
    await cameraManager.load()
    let controllers = await cameraManager.controllers
    let cameras = await withTaskGroup(of: [CameraSnapshot].self) { group in
      for controller in controllers {
        group.addTask {
          let cameras = await controller.cameras
          var cameraSnapshots: [CameraSnapshot] = []
          for camera in cameras {
            await cameraSnapshots.append(
              CameraSnapshot(
                camera: camera,
                name: camera.name,
                zoom: camera.zoom
              )
            )
          }
          
          return cameraSnapshots
        }
      }
      return await group
        .reduce(into: []) { $0 += $1 }
    }
    
    self.cameras = cameras
  }
}

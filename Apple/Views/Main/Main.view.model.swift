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
    cameras = await snapshotCameras()
  }

  func reload() async {
    cameras = []
    await cameraManager.load()
    cameras = await snapshotCameras()
  }

  private func snapshotCameras() async -> [CameraSnapshot] {
    let controllers = await cameraManager.controllers
    return await withTaskGroup(of: [CameraSnapshot].self) { group in
      for controller in controllers {
        group.addTask {
          let cameras = await controller.cameras
          var cameraSnapshots: [CameraSnapshot] = []
          for camera in cameras {
            let isReg = camera is CameraMetaRegistration
            await cameraSnapshots.append(
              CameraSnapshot(
                camera: camera,
                name: camera.name,
                zoom: camera.zoom,
                isRegistration: isReg
              )
            )
          }
          return cameraSnapshots
        }
      }
      return await group
        .reduce(into: []) { $0 += $1 }
    }
  }
}

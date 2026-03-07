//
//  Camera.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import SwiftUI
internal import Combine
import AVFoundation
import CoreImage

public enum CameraError: String {
  case notInit
  case noPermissions
  case noInput
  case noOutput
  case noVideo
  case noSession
  case noDelegate
  
  public var rawValue: String {
    switch self {
      case .notInit:
        return "Not initialized".localizedCapitalized
      case .noPermissions:
        return "Do not have permissions".localizedCapitalized
      case .noInput:
        return "Input stream is not valid".localizedCapitalized
      case .noOutput:
        return "Output stream is not valid".localizedCapitalized
      case .noVideo:
        return "No video feed".localizedCapitalized
      case .noSession:
        return "No feed session".localizedCapitalized
      case .noDelegate:
        return "Missing delegate".localizedCapitalized
    }
  }
}

public enum CameraState: Equatable {
  case disconnected(CameraError?)
  case forceDisconnect
  case disconnecting
  case connecting
  case connected
  case started
  case starting
  case stopped
  case stopping
  
  var stringValue: String {
    switch self {
      case .disconnected(let error):
        return "disconnected - \(error?.rawValue ?? "")".localizedCapitalized
      case .forceDisconnect:
        return "force disconnect"
      case .disconnecting:
        return "disconnecting"
      case .connecting:
        return "Connecting"
      case .connected:
        return "connected"
      case .started:
        return "started"
      case .starting:
        return "starting"
      case .stopped:
        return "stopped"
      case .stopping:
        return "stopping"
    }
  }
}

@MainActor
class CameraSnapshot: Identifiable {
  let id: String = UUID().uuidString
  let name: String
  let zoom: String
  let device: (any Camera)?
  let isRegistration: Bool

  init(
    camera: any Camera,
    name: String,
    zoom: String,
    isRegistration: Bool = false
  ) {
    self.device = camera
    self.name = name
    self.zoom = zoom
    self.isRegistration = isRegistration
  }
  
  init(
    state: CameraState,
    name: String,
    zoom: String
  ) {
    self.device = nil
    self.name = name
    self.zoom = zoom
    self.isRegistration = false
  }
}

public protocol Camera: Actor {
  var state: CameraState { get }
  var name: String {get}
  var zoom: String {get}
  
  func connect(nextFrame: @escaping (CIImage) -> Void) async
  func disconnect() async
  func start() async
  func stop() async
  
  func stateUpdates() -> AsyncStream<CameraState> 
}

//
//  Camera.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import Foundation

public enum CameraError: Equatable {
  case notInit
}

public enum CameraState: Equatable {
  case unavailable(CameraError)
  case disconnecting
  case disconnected(CameraError?)
  case connecting
  case connected
}

public protocol Camera: Actor {
  var state: CameraState { get }
  
  init() async
  
  func connect() async throws
  func disconnect() async throws
  func start() async throws
  func stop() async throws
  func pause() async throws
  func resume() async throws
}

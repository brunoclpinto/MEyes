//
//  Camera.meta.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import Foundation

public actor CameraMeta: Camera {
  public private(set) var state: CameraState = .unavailable(.notInit)
  
  public init() async {
    
  }
  
  public func connect() async throws {
    <#code#>
  }
  
  public func disconnect() async throws {
    <#code#>
  }
  
  public func start() async throws {
    <#code#>
  }
  
  public func stop() async throws {
    <#code#>
  }
  
  public func pause() async throws {
    <#code#>
  }
  
  public func resume() async throws {
    <#code#>
  }
}

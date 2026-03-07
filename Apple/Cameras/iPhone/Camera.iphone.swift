//
//  Camera.iphone.swift
//  MEyes
//
//  Created by Bruno Pinto on 25/02/2026.
//

import AVFoundation
import CoreImage

nonisolated private final class IphoneCameraOutputDelegate: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
  let nextFrame: ((CIImage) -> Void)
  
  init(nextFrame: @escaping (CIImage) -> Void) {
    self.nextFrame = nextFrame
    super.init()
  }
  
  func captureOutput(_ output: AVCaptureOutput,
                     didOutput sampleBuffer: CMSampleBuffer,
                     from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return
    }
    
    nextFrame(CIImage(cvPixelBuffer: pixelBuffer))
  }
}

public actor CameraIphone: Camera {
  public private(set) var state: CameraState = .disconnected(.notInit)
  public let name: String
  public let zoom: String
  
  private let device: AVCaptureDevice
  private var session: AVCaptureSession?
  private var input: AVCaptureDeviceInput?
  private var output: AVCaptureVideoDataOutput?
  private var videoConnection: AVCaptureConnection?
  private var delegate: IphoneCameraOutputDelegate?
  private let sessionQueue = DispatchQueue(label: "camera.session.queue")
  private var stateContinuations: [AsyncStream<CameraState>.Continuation] = []
  
  init(device: AVCaptureDevice) async {
    self.device = device
    self.name = await device.deviceType.humanReadable
    self.zoom = await device.deviceType.zoom
  }
  
  public func connect(nextFrame: @escaping (CIImage) -> Void) async {
    switch state {
      case
          .connected,
          .connecting,
          .disconnecting,
          .started,
          .starting,
          .stopped,
          .stopping,
          .forceDisconnect:
        return
      case
          .disconnected:
        break
    }
    
    setState(.connecting)
    guard await AVCaptureDevice.requestAccess(for: .video) else {
      await forceDisconnect()
      setState(.disconnected(.noPermissions))
      return
    }
    
    let session = AVCaptureSession()
    session.beginConfiguration()
    
    guard
      let input = try? AVCaptureDeviceInput(device: self.device),
      session.canAddInput(input)
    else {
      session.commitConfiguration()
      await forceDisconnect()
      setState(.disconnected(.noInput))
      return
    }
    session.addInput(input)
    
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    ]
    
    let delegate = IphoneCameraOutputDelegate(nextFrame: nextFrame)
    let outputQueue = DispatchQueue(label: "camera.stream.video.output.queue")
    output.setSampleBufferDelegate(delegate, queue: outputQueue)
    
    guard session.canAddOutput(output) else {
      session.commitConfiguration()
      await forceDisconnect()
      setState(.disconnected(.noOutput))
      return
    }
    session.addOutput(output)
    
    guard let connection = output.connection(with: .video) else {
      session.commitConfiguration()
      await forceDisconnect()
      setState(.disconnected(.noVideo))
      return
    }
    connection.videoRotationAngle = 90
    
    session.commitConfiguration()
    
    self.session = session
    self.input = input
    self.output = output
    self.videoConnection = connection
    self.delegate = delegate
    
    setState(.connected)
  }
  
  public func disconnect() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected:
        return
      case
          .started:
        _ = await stop()
        break
      case
          .forceDisconnect,
          .stopped,
          .connected:
        break
    }
    setState(.disconnecting)
    
    guard let session = self.session else {
      setState(.disconnected(.noSession))
      return
    }
    session.beginConfiguration()
    
    if let output {
      output.setSampleBufferDelegate(nil, queue: nil)
    }
    
    if let output = self.output { session.removeOutput(output) }
    if let input = self.input { session.removeInput(input) }
    
    session.commitConfiguration()
    
    self.videoConnection = nil
    self.output = nil
    self.input = nil
    self.delegate = nil
    self.session = nil
    
    setState(.disconnected(nil))
  }
  
  public func start() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected(_),
          .forceDisconnect,
          .started:
        return
      case
          .connected,
          .stopped:
        break
    }
    
    setState(.starting)
    
    guard let session else {
      await forceDisconnect()
      setState(.disconnected(.noSession))
      return
    }
    guard delegate != nil else {
      await forceDisconnect()
      setState(.disconnected(.noDelegate))
      return
    }
    if !session.isRunning {
      session.startRunning()
    }
    
    setState(.started)
  }
  
  public func stop() async {
    switch state {
      case
          .connecting,
          .disconnecting,
          .starting,
          .stopping,
          .disconnected(_),
          .forceDisconnect,
          .stopped,
          .connected:
        return
      case
          .started:
        break
    }
    setState(.stopping)
    
    guard let session else {
      await forceDisconnect()
      setState(.disconnected(.noSession))
      return
    }
    
    guard delegate != nil else {
      await forceDisconnect()
      setState(.disconnected(.noDelegate))
      return
    }
    
    if session.isRunning {
      session.stopRunning()
    }
    
    setState(.stopped)
  }
}

/// Publisher
public extension CameraIphone {
  func stateUpdates() -> AsyncStream<CameraState> {
    AsyncStream { cont in
      stateContinuations.append(cont)
      cont.yield(state) // initial state
    }
  }
  
  private func setState(_ newState: CameraState) {
    state = newState
    stateContinuations.forEach { $0.yield(newState) }
  }
}

/// Support
private extension CameraIphone {
  private func forceDisconnect() async {
    state = .forceDisconnect
    _ = await disconnect()
  }
}

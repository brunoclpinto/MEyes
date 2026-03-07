//
//  Camera.view.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI

struct CameraView: View {
  @StateObject var viewModel: CameraViewModel
  @Environment(\.dismiss) private var dismiss
  
  var body: some View {
    VStack {
      Text("\(viewModel.camera.name)")
        .font(.title)
      Spacer()
        .frame(height: 20)

      HStack {
        Spacer()
        switch viewModel.action {
          case .button(let icon, let label, let hint):
            Button {
              Task {
                await viewModel.performAction()
              }
            } label: {
              Image(systemName: icon)
                .font(.title)
                .padding(50)
                .background(Circle().fill(Color(.darkGray).opacity(0.8)))
                .imageScale(.large)
            }
            .buttonStyle(.plain)
            .accessibilityLabel(label)
            .accessibilityHint(hint)
          case .status(let message):
            VStack(spacing: 12) {
              ProgressView()
                .progressViewStyle(.circular)
                .scaleEffect(2)
              Text(message)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            }
            .padding(30)
        }
        Spacer()
      }
    }
    .frame(maxHeight: .infinity, alignment: .top)
    .onAppear {
      Task {
        await viewModel.startObservingState()
        guard
          let camera = viewModel.camera.device
        else {
          return
        }
        await camera.connect { [weak viewModel] image in
          guard let viewModel else { return }
          Task {
            await viewModel.processFrame(image)
          }
        }
      }
    }
    .onDisappear {
      Task {
        await viewModel.camera.device?.disconnect()
        viewModel.stopObservingState()
      }
    }
    .onChange(of: viewModel.state) {
      guard
        viewModel.camera.isRegistration,
        viewModel.state == .started
      else {
        return
      }
      
      dismiss()
    }
  }
}

#Preview {
  CameraView(
    viewModel: CameraViewModel(
      camera: CameraSnapshot(
        state: .connected,
        name: "Camera",
        zoom: ""
      )
    )
  )
}

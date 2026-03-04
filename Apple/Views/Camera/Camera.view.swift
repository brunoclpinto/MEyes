//
//  Camera.view.swift
//  MEyes
//
//  Created by Bruno Pinto on 27/02/2026.
//

import SwiftUI

struct CameraView: View {
  @StateObject var viewModel: CameraViewModel
  
  var body: some View {
    VStack {
      Text("\(viewModel.camera.name)")
        .font(.title)
      Spacer()
        .frame(height: 20)
      HStack {
        Spacer()
        Button {
          Task {
            await viewModel.performActionButton()
          }
        } label: {
          Image(systemName: viewModel.actionButtonIcon.rawValue)
            .font(.title)
            .padding(50)
            .background(Circle().fill(Color(.darkGray).opacity(0.8)))
            .imageScale(.large)
        }
        .disabled(!viewModel.actionButtonEnabled)
        .buttonStyle(.plain)
        .accessibilityLabel(viewModel.actionButtonIcon.accessibleTitle)
        .accessibilityHint(viewModel.actionButtonIcon.accessibleTitle)
        Spacer()
      }
      Text(viewModel.state.stringValue)
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
        await camera.connect { image in
        }
      }
    }
    .onDisappear {
      Task {
        await viewModel.camera.device?.disconnect()
        viewModel.stopObservingState()
      }
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

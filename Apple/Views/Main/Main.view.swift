//
//  ContentView.swift
//  MEyes
//
//  Created by Bruno Pinto on 23/02/2026.
//

import SwiftUI

struct MainView: View {
  @StateObject var viewModel = MainViewModel()
  @State private var needsReload = false

  var body: some View {
    Group() {
      if viewModel.cameras.isEmpty {
        ProgressView()
          .progressViewStyle(.circular)
          .scaleEffect(5)
        Text("Finding cameras")
      } else {
        NavigationStack {
          VStack {
            ForEach(viewModel.cameras) { camera in
              NavigationLink {
                CameraView(
                  viewModel: CameraViewModel(
                    camera: camera
                  )
                )
                .onDisappear {
                  if camera.isRegistration {
                    needsReload = true
                  }
                }
              } label: {
                HStack(alignment: .center, spacing: 2) {
                  Text(camera.name).font(.title)
                  if !camera.zoom.isEmpty {
                    Text(camera.zoom).font(.callout)
                  }
                }
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .center)
              }
              .buttonStyle(.bordered)
              .background(Color(.darkGray).opacity(0.2))
              .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
              .foregroundStyle(.primary)
            }
          }
          .frame(maxHeight: .infinity, alignment: .top)
          .onChange(of: needsReload) {
            guard needsReload else { return }
            needsReload = false
            Task {
              await viewModel.reload()
            }
          }
        }
      }
    }
    .task {
      await viewModel.load()
    }
  }
}

#Preview {
  MainView()
}

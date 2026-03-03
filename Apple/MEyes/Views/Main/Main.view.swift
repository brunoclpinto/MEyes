//
//  ContentView.swift
//  MEyes
//
//  Created by Bruno Pinto on 23/02/2026.
//

import SwiftUI

struct MainView: View {
  @StateObject var viewModel = MainViewModel()
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
              } label: {
                HStack(alignment: .center, spacing: 2) {
                  Text(camera.name).font(.title)
                  Text(camera.zoom).font(.callout)
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

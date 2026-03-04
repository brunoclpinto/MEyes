// Zero domain knowledge — no bus types, no CoreML, no Vision imports.

// MARK: - AnyFlow

/// Type-erased wrapper so WorkflowManager can hold heterogeneous flows.
public struct AnyFlow<Input: Sendable, Output: Sendable>: Sendable {
    public let id: String
    private let _run: @Sendable (Input) async throws -> Output

    public init<F: Flow>(_ flow: F) where F.Input == Input, F.Output == Output {
        self.id = flow.id
        self._run = { try await flow.run(input: $0) }
    }

    public func run(input: Input) async throws -> Output {
        try await _run(input)
    }
}

// MARK: - WorkflowManager

/// Async execution engine: runs concurrent groups in parallel via TaskGroup,
/// then threads typed output to serial dependents.
/// WorkflowManager has zero domain knowledge.
public final class WorkflowManager<SharedInput: Sendable>: Sendable {

    public init() {}

    /// Run a concurrent group of flows that all take `SharedInput`, collecting results.
    /// Returns a WorkflowOutput with each flow's output stored under its ID.
    public func runConcurrent<Output: Sendable>(
        flows: [AnyFlow<SharedInput, Output>],
        input: SharedInput
    ) async -> WorkflowOutput {
        var out = WorkflowOutput()
        await withTaskGroup(of: (String, Output?).self) { group in
            for flow in flows {
                group.addTask {
                    do {
                        let result = try await flow.run(input: input)
                        return (flow.id, result)
                    } catch {
                        return (flow.id, nil)
                    }
                }
            }
            for await (id, result) in group {
                if let result {
                    out.set(result, for: id)
                }
            }
        }
        return out
    }

    /// Run a single serial flow whose input is derived from a previous WorkflowOutput.
    public func runSerial<UpstreamOutput: Sendable, FlowInput: Sendable, Output: Sendable>(
        flow: AnyFlow<FlowInput, Output>,
        upstream: WorkflowOutput,
        upstreamId: String,
        transform: @Sendable (UpstreamOutput) -> FlowInput?
    ) async -> (output: Output?, workflowOutput: WorkflowOutput) {
        var out = upstream
        guard let upstreamValue = upstream.get(UpstreamOutput.self, for: upstreamId),
              let flowInput = transform(upstreamValue) else {
            return (nil, out)
        }
        do {
            let result = try await flow.run(input: flowInput)
            out.set(result, for: flow.id)
            return (result, out)
        } catch {
            return (nil, out)
        }
    }
}

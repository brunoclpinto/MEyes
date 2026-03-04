// MARK: - FlowStatus

public enum FlowStatus {
    case success
    case failed(Error)
    case skipped
}

// MARK: - FlowResult

public struct FlowResult<Output> {
    public let flowId: String
    public let status: FlowStatus
    public let output: Output?

    public init(flowId: String, status: FlowStatus, output: Output?) {
        self.flowId = flowId
        self.status = status
        self.output = output
    }
}

// MARK: - Flow

/// A single processing unit with typed Input and Output.
public protocol Flow<Input, Output>: Sendable {
    associatedtype Input: Sendable
    associatedtype Output: Sendable

    var id: String { get }
    func run(input: Input) async throws -> Output
}

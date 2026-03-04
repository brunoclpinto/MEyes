// MARK: - WorkflowGraph

/// Declares the topology of a workflow: concurrent groups and serial dependencies.
/// Zero domain knowledge — describes structure only.
public struct WorkflowGraph: Sendable {

    /// A set of flow IDs that can execute concurrently (they share the same input).
    public struct ConcurrentGroup: Sendable {
        public let flowIds: [String]
        public init(_ flowIds: [String]) { self.flowIds = flowIds }
    }

    /// A flow that depends on a specific upstream flow's output.
    public struct SerialStep: Sendable {
        /// The flow to run.
        public let flowId: String
        /// The ID of the upstream flow whose output feeds into this flow.
        public let dependsOn: String

        public init(flowId: String, dependsOn: String) {
            self.flowId = flowId
            self.dependsOn = dependsOn
        }
    }

    public let concurrentGroup: ConcurrentGroup
    public let serialSteps: [SerialStep]

    public init(concurrentGroup: ConcurrentGroup, serialSteps: [SerialStep] = []) {
        self.concurrentGroup = concurrentGroup
        self.serialSteps = serialSteps
    }
}

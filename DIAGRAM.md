```mermaid
graph TD
    %% Styling
    classDef manager fill:#f9f,stroke:#333,stroke-width:2px,color:#000
    classDef component fill:#bbf,stroke:#333,stroke-width:1px,color:#000
    classDef model fill:#bfb,stroke:#333,stroke-width:1px,color:#000
    classDef dataset fill:#fbb,stroke:#333,stroke-width:1px,color:#000
    classDef partitioner fill:#ffb,stroke:#333,stroke-width:1px,color:#000
    classDef spacing fill:none,stroke:none

    %% Main Components
    ExperimentMgmt[ExperimentManager]:::manager
    DeviceMgmt[DeviceManager]:::manager
    DataComp[DataCompression]:::component
    LogMgmt[LogManager]:::component
    MasterDict[MasterDict]:::component
    NetworkClient[NetworkClient]:::component
    RemoteConn[RemoteConnection]:::component
    PowerMon[PowerMonitor]:::component

    %% Model Components
    BaseModel[BaseModel]:::model
    WrappedModel[WrappedModel]:::model
    ModelRegistry[ModelRegistry]:::model
    CustomModel[CustomModel]:::model

    %% Dataset Components
    BaseDataset[BaseDataset]:::dataset
    ImageNetDataset[ImageNetDataset]:::dataset
    OnionDataset[OnionDataset]:::dataset
    DataManager[DataManager]:::dataset

    %% Partitioner Components
    Partitioner[Partitioner]:::partitioner
    CyclePartitioner[CyclePartitioner]:::partitioner
    RegressionPartitioner[RegressionPartitioner]:::partitioner

    %% Subgraphs with styling
    subgraph API_Layer["API Layer"]
        direction LR
        ExperimentMgmt --> DeviceMgmt
        ExperimentMgmt --> DataComp
        ExperimentMgmt --> LogMgmt
        ExperimentMgmt --> MasterDict
        ExperimentMgmt --> NetworkClient
        ExperimentMgmt --> PowerMon
        DeviceMgmt --> RemoteConn
        NetworkClient --> DataComp
    end

    subgraph Model_Layer["Model Layer"]
        direction LR
        BaseModel --> WrappedModel
        ModelRegistry --> CustomModel
        ModelRegistry --> BaseModel
    end

    subgraph Data_Layer["Data Layer"]
        direction LR
        BaseDataset --> ImageNetDataset
        BaseDataset --> OnionDataset
        DataManager --> BaseDataset
    end

    subgraph Partitioning_Layer["Partitioning Layer"]
        direction LR
        Partitioner --> CyclePartitioner
        Partitioner --> RegressionPartitioner
    end

    %% Layer styling
    style API_Layer fill:#f9f3,stroke:#333,stroke-width:2px,color:#000
    style Model_Layer fill:#bfb3,stroke:#333,stroke-width:2px,color:#000
    style Data_Layer fill:#fbb3,stroke:#333,stroke-width:2px,color:#000
    style Partitioning_Layer fill:#ffb3,stroke:#333,stroke-width:2px,color:#000

    %% Vertical spacing connections
    SpacingNode1[" "]:::spacing
    SpacingNode2[" "]:::spacing
    SpacingNode3[" "]:::spacing
    SpacingNode4[" "]:::spacing
    
    API_Layer --> SpacingNode1
    SpacingNode1 --> Model_Layer
    Model_Layer --> SpacingNode2
    SpacingNode2 --> Data_Layer
    Data_Layer --> SpacingNode3
    SpacingNode3 --> Partitioning_Layer
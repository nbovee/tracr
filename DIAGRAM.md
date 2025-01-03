```mermaid
graph TD
    %% Styling
    classDef manager fill:#f9f,stroke:#333,stroke-width:2px
    classDef component fill:#bbf,stroke:#333,stroke-width:1px
    classDef model fill:#bfb,stroke:#333,stroke-width:1px
    classDef dataset fill:#fbb,stroke:#333,stroke-width:1px
    classDef partitioner fill:#ffb,stroke:#333,stroke-width:1px

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

    %% Add invisible nodes for vertical alignment
    InvisibleNode1[" "]
    InvisibleNode2[" "]
    InvisibleNode3[" "]
    
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
    style API_Layer fill:#f9f3,stroke:#333,stroke-width:2px
    style Model_Layer fill:#bfb3,stroke:#333,stroke-width:2px
    style Data_Layer fill:#fbb3,stroke:#333,stroke-width:2px
    style Partitioning_Layer fill:#ffb3,stroke:#333,stroke-width:2px
    
    %% Hide invisible nodes
    style InvisibleNode1 fill:none,stroke:none
    style InvisibleNode2 fill:none,stroke:none
    style InvisibleNode3 fill:none,stroke:none
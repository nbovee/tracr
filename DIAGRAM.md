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

    %% Relationships
    ExperimentMgmt --> |uses| DeviceMgmt
    ExperimentMgmt --> |uses| DataComp
    ExperimentMgmt --> |uses| LogMgmt
    ExperimentMgmt --> |uses| MasterDict
    ExperimentMgmt --> |uses| NetworkClient
    ExperimentMgmt --> |uses| PowerMon

    DeviceMgmt --> |uses| RemoteConn
    NetworkClient --> |uses| DataComp

    BaseModel --> |extends| WrappedModel
    ModelRegistry --> |registers| CustomModel
    ModelRegistry --> |manages| BaseModel

    BaseDataset --> |extends| ImageNetDataset
    BaseDataset --> |extends| OnionDataset
    DataManager --> |manages| BaseDataset

    Partitioner --> |extends| CyclePartitioner
    Partitioner --> |extends| RegressionPartitioner

    %% Subgraphs with styling
    subgraph API_Layer["API Layer"]
        direction TB
        ExperimentMgmt
        DeviceMgmt
        DataComp
        LogMgmt
        MasterDict
        NetworkClient
        RemoteConn
        PowerMon
    end

    subgraph Model_Layer["Model Layer"]
        direction TB
        BaseModel
        WrappedModel
        ModelRegistry
        CustomModel
    end

    subgraph Data_Layer["Data Layer"]
        direction TB
        BaseDataset
        ImageNetDataset
        OnionDataset
        DataManager
    end

    subgraph Partitioning_Layer["Partitioning Layer"]
        direction TB
        Partitioner
        CyclePartitioner
        RegressionPartitioner
    end

    %% Layer styling
    style API_Layer fill:#f9f3,stroke:#333,stroke-width:2px
    style Model_Layer fill:#bfb3,stroke:#333,stroke-width:2px
    style Data_Layer fill:#fbb3,stroke:#333,stroke-width:2px
    style Partitioning_Layer fill:#ffb3,stroke:#333,stroke-width:2px
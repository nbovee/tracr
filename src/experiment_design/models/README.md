# Model Hooking Mechanism

## Overview

The model hooking system provides a flexible framework for analyzing neural network performance and enabling split computing. By strategically inserting hooks at each layer of a neural network, this system can:

1. Collect detailed performance metrics (timing, energy consumption, memory usage)
2. Capture intermediate outputs at any layer
3. Enable controlled execution to arbitrary split points
4. Facilitate distributed model inference across devices

This README explains the technical implementation of the hooking mechanism, focusing on how hooks are registered, how execution flow is controlled, and how metrics are collected.

## Hook Registration Flow

The hook registration process follows these steps:

```
┌──────────────────────────┐
│                          │
│     Model Initialization │
│                          │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│                          │
│    Analyze Model Layers  │<──────┐
│                          │       │
└─────────────┬────────────┘       │
              │                    │
              ▼                    │
┌──────────────────────────┐       │
│                          │       │
│  Is Current Layer a      │       │
│  Container Module?       │───Yes─┘
│                          │
└─────────────┬────────────┘
              │
              │ No
              ▼
┌──────────────────────────┐
│                          │
│  Initialize Metrics      │
│  Storage for Layer       │
│                          │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│                          │
│  Register Pre-Hook       │
│  for Layer               │
│                          │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│                          │
│  Register Post-Hook      │
│  for Layer               │
│                          │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│                          │
│  More Layers to Process? │───Yes──┐
│                          │        │
└─────────────┬────────────┘        │
              │                     │
              │ No                  │
              ▼                     │
┌──────────────────────────┐        │
│                          │        │
│  Perform Warmup Passes   │        │
│                          │        │
└─────────────┬────────────┘        │
              │                     │
              ▼                     │
┌──────────────────────────┐        │
│                          │        │
│  Model Ready for Use     │        │
│                          │        │
└──────────────────────────┘        │
                                    │
                                    │
                                    ▼
                           ┌──────────────────────┐
                           │                      │
                           │  Next Layer          │
                           │                      │
                           └──────────────────────┘
```

## Execution Flow with Hooks

When the model is executed, the following process occurs:

```
┌──────────────────┐                      ┌──────────────────┐
│                  │                      │                  │
│     Forward      │                      │     Forward      │
│     Pre-Hook     │                      │     Post-Hook    │
│                  │                      │                  │
└───────┬──────────┘                      └────────┬─────────┘
        │                                          │
        │                                          │
        ▼                                          │
┌──────────────────┐      ┌───────────────┐        │
│                  │      │               │        │
│  Start Layer     │ ─────► Layer Module  │────────┘
│  Measurement     │      │ Computation   │
│                  │      │               │
└──────────────────┘      └───────────────┘
                                 │
                                 │
                                 ▼
                          ┌──────────────────┐
                          │                  │
                          │  End Layer       │
                          │  Measurement     │
                          │                  │
                          └────────┬─────────┘
                                   │
                                   │
                                   ▼
                          ┌──────────────────┐
                          │                  │
                          │  Bank Layer      │
                          │  Output          │
                          │                  │
                          └────────┬─────────┘
                                   │
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ Split Point      │
                          │ Reached?         │───No───┐
                          │                  │        │
                          └────────┬─────────┘        │
                                   │                  ▼
                                   │           ┌────────────────┐
                                  Yes          │                │
                                   │           │ Continue to    │
                                   ▼           │ Next Layer     │
                          ┌──────────────────┐ │                │
                          │                  │ └────────────────┘
                          │ Raise Hook Exit  │
                          │ Exception        │
                          │                  │
                          └────────┬─────────┘
                                   │
                                   │
                                   ▼
                          ┌──────────────────┐
                          │                  │
                          │ Return Early     │
                          │ Output           │
                          │                  │
                          └──────────────────┘
```

## Technical Implementation

### 1. Hook Creation and Registration

The `WrappedModel._walk_modules` method traverses the model hierarchy and registers hooks on each computational layer:

```python
def _walk_modules(self, modules: Any, depth: int, walk_i: int) -> int:
    """Traverse model hierarchy recursively to register hooks on appropriate layers."""
    for child in modules:
        child_name = child.__class__.__name__
        children = list(child.children())

        if children and depth < self.depth:
            # Skip container modules, go deeper in the hierarchy
            walk_i = self._walk_modules(children, depth + 1, walk_i)
        elif isinstance(child, torch.nn.Module):
            # Register hooks on computational layers
            walk_i = self._register_layer(child, child_name, depth, walk_i)

    return walk_i
```

For each computational layer, pre-hooks and post-hooks are registered:

```python
def _register_layer(self, layer: torch.nn.Module, layer_name: str, depth: int, walk_i: int) -> int:
    """Register hooks and initialize metrics storage for a single model layer."""
    # Initialize metrics storage for this layer
    self.forward_info[walk_i] = copy.deepcopy(LAYER_TEMPLATE)
    self.forward_info[walk_i].update({
        "layer_id": walk_i,
        "layer_type": layer_info.class_name,
        "output_bytes": layer_info.output_bytes,
        "inference_time": None,
    })
    
    # Register pre-hook for timing and input control
    self.forward_hooks.append(
        layer.register_forward_pre_hook(
            create_forward_prehook(self, walk_i, layer_name, (0, 0), self.device)
        )
    )
    
    # Register post-hook for metrics collection and execution control
    self.forward_post_hooks.append(
        layer.register_forward_hook(
            create_forward_posthook(self, walk_i, layer_name, (0, 0), self.device)
        )
    )
    
    walk_i += 1
    return walk_i
```

### 2. Pre-Hook Implementation

The pre-hook is responsible for initialization and timing:

```python
def create_forward_prehook(wrapped_model, layer_index, layer_name, input_shape, device) -> Callable:
    """Create pre-hook for layer measurement and input modification."""
    
    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        """Execute pre-module operations for timing and input processing."""
        hook_output = layer_input

        # Edge device mode initialization
        if wrapped_model.start_i == 0:
            if layer_index == 0:
                # Initialize output storage
                wrapped_model.banked_output = {}
                
                # Start global energy monitoring if enabled
                if wrapped_model.collect_metrics and wrapped_model.metrics_collector:
                    wrapped_model.metrics_collector.set_split_point(wrapped_model.stop_i)
                    wrapped_model.metrics_collector.start_global_measurement()
            
            # Start layer-specific metrics collection
            if wrapped_model.collect_metrics and wrapped_model.metrics_collector:
                wrapped_model.metrics_collector.start_layer_measurement(layer_index)
        
        # Cloud device mode - override input with banked output from edge
        else:
            if layer_index == 0:
                wrapped_model.banked_output = layer_input[0]()
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)
        
        # Record layer start time
        if wrapped_model.log and wrapped_model.collect_metrics:
            start_time = time.perf_counter()
            wrapped_model.layer_times[layer_index] = start_time
        
        return hook_output
    
    return pre_hook
```

### 3. Post-Hook Implementation

The post-hook collects metrics and controls execution flow:

```python
def create_forward_posthook(wrapped_model, layer_index, layer_name, input_shape, device) -> Callable:
    """Create post-hook for metrics collection and execution control."""
    
    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-module operations for metrics and execution control."""
        
        # Collect metrics if enabled
        if wrapped_model.log and wrapped_model.collect_metrics:
            layer_data = wrapped_model.forward_info.get(layer_index, {})
            
            # Use metrics collector if available
            if wrapped_model.metrics_collector:
                wrapped_model.metrics_collector.end_layer_measurement(layer_index, output)
                
                # Update layer metrics dictionary
                if layer_index in wrapped_model.metrics_collector.layer_metrics:
                    wrapped_model.forward_info[layer_index].update(
                        wrapped_model.metrics_collector.layer_metrics[layer_index]
                    )
            else:
                # Fallback to direct timing measurement
                if layer_index in wrapped_model.layer_times:
                    end_time = time.perf_counter()
                    start_time = wrapped_model.layer_times[layer_index]
                    elapsed_time = end_time - start_time
                    wrapped_model.forward_info[layer_index]["inference_time"] = elapsed_time
        
        # Handle output banking and early exit
        if wrapped_model.start_i == 0:  # Edge device mode
            prepare_exit = wrapped_model.stop_i <= layer_index
            
            # Save output if this is a layer we want to keep or if we're at the split point
            if layer_index in wrapped_model.save_layers or prepare_exit:
                wrapped_model.banked_output[layer_index] = output
            
            # Exit at split point with collected outputs
            if prepare_exit:
                raise HookExitException(wrapped_model.banked_output)
        else:  # Cloud device mode
            # Replace output with banked value if available
            if layer_index in wrapped_model.banked_output:
                output = wrapped_model.banked_output[layer_index]
        
        return output
    
    return post_hook
```

### 4. Early Exit Mechanism

The `HookExitException` and `EarlyOutput` classes facilitate controlled model splitting:

```python
class HookExitException(Exception):
    """Exception used to halt model execution at a designated layer."""
    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result

@dataclass
class EarlyOutput:
    """Container for intermediate outputs from a partial model execution."""
    inner_dict: Union[Dict[str, Any], Tensor]
    
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Dict[str, Any], Tensor]:
        """Return inner dictionary or tensor when called."""
        return self.inner_dict
```

### 5. Forward Pass Control Flow

The `forward` method in `WrappedModel` controls the execution flow:

```python
def forward(self, x: Union[torch.Tensor, Image.Image], 
            inference_id: Optional[str] = None,
            start: int = 0, 
            end: Union[int, float] = np.inf,
            log: bool = True) -> Any:
    """Execute model forward pass with configurable start/end layers."""
    
    # Configure forward pass parameters
    self.log = log
    self.start_i = start
    self.stop_i = end if end != np.inf else self.layer_count
    self._setup_inference_id(inference_id)
    
    try:
        # Execute forward pass normally
        output = self._execute_forward(x)
    except HookExitException as e:
        # Handle early exit at split point
        output = self._handle_early_exit(e)
    
    # Process and store results
    self._handle_results(start_time)
    return output
```

## Dual Mode Operation

The hooking system enables the model to operate in two distinct modes:

### 1. Edge Device Mode (`start_i = 0`)

When running in edge device mode:
- The model processes from the first layer up to the specified split point
- At the split point, execution is halted via `HookExitException`
- Intermediate outputs are captured and returned as `EarlyOutput`
- Metrics are collected for each layer up to the split point

### 2. Cloud Device Mode (`start_i > 0`)

When running in cloud device mode:
- The first layer's input is replaced with the intermediate output from the edge device
- Processing continues from the split point to the end of the model
- Metrics are collected for the remainder of the model
- The final output is returned

## Metrics Collection

The hooking system integrates with a comprehensive metrics collection framework:

1. **Timing Metrics**: Measure execution time for each layer and the full model
2. **Energy Metrics**: Track power consumption and energy usage
3. **Memory Metrics**: Monitor memory allocation and utilization
4. **Resource Metrics**: Measure CPU, GPU, and memory utilization

For each layer, metrics are stored in a standardized format:

```python
{
    "layer_id": layer_id,
    "layer_type": layer_type,
    "output_bytes": size_in_bytes,
    "inference_time": execution_time,
    "processing_energy": energy_consumed,
    "power_reading": current_power,
    "gpu_utilization": gpu_usage_percent,
    "memory_utilization": memory_usage_percent,
    "cpu_utilization": cpu_usage_percent,
    "total_energy": total_energy_consumed,
}
```

## Performance Considerations

Several factors affect the performance of the hooking mechanism:

1. **Hook Depth**: 
   - Deeper hook registration (traversing more module levels) provides more granular metrics
   - But also increases overhead and can slow down execution

2. **Metrics Collection**: 
   - Enable only necessary metrics to reduce overhead
   - Power monitoring can be particularly resource-intensive

3. **Split Point Selection**:
   - Choose split points at natural boundaries in the model architecture
   - Consider both computational and memory requirements at each potential split point

4. **Warmup Iterations**:
   - Always perform warmup iterations to stabilize timing measurements
   - Initial runs may have higher variance due to JIT compilation and memory allocation

## Usage Examples

### Basic Model Analysis

```python
from src.experiment_design.models.model_hooked import WrappedModel

# Create wrapped model
model = WrappedModel(config, master_dict)

# Run full model with metrics collection
output = model.forward(input_tensor, log=True)

# Get layer metrics
metrics = model.get_layer_metrics()
```

### Split Computing Setup

```python
# Edge device: Run model up to layer 10
edge_output = edge_model.forward(input_tensor, end=10)

# Transfer edge_output to cloud device

# Cloud device: Continue from layer 10
final_output = cloud_model.forward(edge_output, start=10)
```

### Analyzing Specific Layers

```python
# Run model and store outputs from specific layers
model.save_layers = {5, 10, 15}  # Store outputs from layers 5, 10, and 15
output = model.forward(input_tensor)

# Access intermediate outputs
layer5_output = model.banked_output[5]
layer10_output = model.banked_output[10]
layer15_output = model.banked_output[15]
```

## Extensions and Customization

The hooking mechanism is designed to be extensible. You can customize it by:

1. Creating specialized hooks for specific layer types
2. Adding new metrics collectors for different hardware platforms
3. Implementing custom splitting strategies for specific model architectures
4. Extending the metrics collection to include additional measurements

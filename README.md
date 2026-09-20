# String Stability Verification

Research code for learning and verifying neural certificates for string stability in interconnected systems. The implementation combines neural vector Lyapunov functions, controller learning, and counterexample-guided verification using Marabou.

Experiments cover vehicle platoons, aircraft systems, and microgrids.

## Repository Structure

| Directory | Experiments |
| --- | --- |
| `stabe_platoon_additive/` | Vehicle platoons with additive disturbances |
| `stabe_platoon_control_performance/` | Vehicle platoon control performance |
| `stabe_platoon_large_scale/` | Large-scale vehicle platoons |
| `stable_aircraft_additive/` | Aircraft systems with additive disturbances |
| `stable_aircraft_control_performance/` | Aircraft control performance |
| `stable_aircraft_large_scale/` | Large-scale aircraft systems |
| `stable_microgrid_control_performance/` | Microgrid control performance |
| `stable_microgrid_large_scale/` | Large-scale microgrids |

The `stabe_` spelling matches the existing directory names.

## Dependencies

Core dependencies include Python, PyTorch, Lightning, NumPy, ONNX, ONNX Simplifier, and [Marabou](https://github.com/NeuralNetworkVerification/Marabou).

```bash
pip install torch lightning numpy onnx onnxsim
```

Install Marabou separately and ensure that `maraboupy` is importable. Update the local Marabou path in `queries_comb.py` if needed. Additional dependencies may be required for individual visualization or pretraining scripts.

## Usage

Clone the repository and run scripts from the corresponding experiment directory:

```bash
git clone https://github.com/JingyuanZhou/String_Stability_Verification.git
cd String_Stability_Verification/stabe_platoon_control_performance
python run_cluster_exp_comb.py
```

Before running, check the system parameters, training settings, pretrained checkpoint paths, and output directories in the scripts.

The main scripts are:

- `run_cluster_exp_comb.py`: experiment configuration and training/verification workflow.
- `training_exp_comb.py`: initial training and counterexample-guided retraining.
- `networks.py`: neural network definitions.
- `generate_combined_model_torch_comb.py`: combined model construction and ONNX export.
- `queries_comb.py`: Marabou verification queries.
- `visualization_training.py` and `visualization_trajectory.py`: result visualization.

**Note:** In the platoon control-performance example, `mode = 0` selects sISS and `mode = 1` selects compositional ISS. With `mode = 1`, the current script exits after training. Use `mode = 0` to continue to the verification stage.

## Contact

For questions, please open an issue in this repository.

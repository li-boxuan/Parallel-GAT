# PPI model

gat_ppi_model.json (compressed in gat_ppi_model.tar) is from [pytorch-GAT](https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(PPI).ipynb) repository, with
'num_of_layers': 3, 'num_heads_per_layer': [4, 4, 6], 'num_features_per_layer': [50, 256, 256, 121]

The model json file is exported using the following code:

```python
model_name=r'models/binaries/gat_PPI_000000.pth'
model_state = torch.load(model_name, map_location=torch.device("cpu"))
param_dict = dict()
for key, value in model_state["state_dict"].items():
    param_dict[key] = value.cpu().detach().numpy().tolist()
text_file = open("./gat_ppi_model.json", "w")
text_file.write(json.dumps(param_dict, indent=4))
text_file.close()
```

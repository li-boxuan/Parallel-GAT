# PPI model

1 layer, 8 heads, feature_dim = 50, output_dim = 121

8 lines (8 x 121): scoring_fn_source
8 lines (8 x 121): scoring_fn_target
8 heads:
   121 lines (121 x 50): linear_proj

The model json file is exported using the following code:

```python
model_name=r'models/binaries/gat_PPI_000000.pth'
model_state = torch.load(model_name, map_location=torch.device("cpu"))
# source: 8 (num_heads) x out_dim (121)
model_file = open("./gat_ppi_model.txt", "w")
param_dict = dict()
for key, value in model_state["state_dict"].items():
   param_dict[key] = value.cpu().detach().numpy().tolist()
source_params = param_dict["gat_net.0.scoring_fn_source"][0]
for row in source_params:
   line = ""
   for elem in row:
      if line == "":
         line = str(elem)
      else:
         line += " " + str(elem)
   line += "\n"
   model_file.write(line)
# target: 8 (num_heads) x out_dim (121)
target_params = param_dict["gat_net.0.scoring_fn_target"][0]
for row in target_params:
   line = ""
   for elem in row:
      if line == "":
         line = str(elem)
      else:
         line += " " + str(elem)
   line += "\n"
   model_file.write(line)

# linear_proj: 968 (121 output_dim * 8 num_heads) x 50 (input_dim)
linear_proj = param_dict["gat_net.0.linear_proj.weight"]
for row in linear_proj:
   line = ""
   for elem in row:
      if line == "":
         line = str(elem)
      else:
         line += " " + str(elem)
   line += "\n"
   model_file.write(line)
model_file.close()
```

# 调用单个大模型
prompt = f"""
You are an expert configuration dependency analyzer for a system with APIs and Infos used to process data.

System Description:
- APIs: Represent external interfaces. They take specific 'inputs' and produce 'outputs'.
- Infos: Represent either starting data points or data processing/calculation logic.
    - Infos with `type: 'input'` are the necessary starting points for a process. They provide initial data specified in their 'outputs'.
    - Other Infos might have 'dependencies' on APIs or other Infos, meaning they need data from those dependencies to perform their logic (often described in their 'name' or implied by the overall goal). They produce data specified in their 'outputs'.
- Your task is to analyze the User Target Description and determine if it can be achieved by chaining the available APIs and Infos. If possible, map the exact dependency flow required.

Available Configurations:

APIs:
```json
{apis_json_string}
```

Infos:
```json
{infos_json_string}
```

User Target Description:
"{target_description}"

Analysis Task:

Identify the essential starting 'input' Info configurations needed based on the target description.
Determine the sequence of APIs and Infos required to process the data from the inputs to achieve the final target. Match 'outputs' from one step to the requirements (explicit 'inputs' or inferred needs) of the next.
Define a final "target" node that represents the calculation or result described in the user target.
Map the dependencies between all required configurations, including the final target node.
Output Format:
Respond ONLY with a single, valid JSON object. Do not include any text before or after the JSON object. The JSON object must strictly adhere to the following structure:


```json
{{
  "success": boolean, // True if the target can be achieved with the given configurations, False otherwise.
  "reasoning": string, // A step-by-step explanation of the dependency path if successful, or a clear reason why the target cannot be achieved.
  "start_node_ids": [string], // List of prefixed IDs (e.g., "info_1", "info_3") corresponding to the REQUIRED 'input' type Info configurations identified in step 1.
  "target_node_name": string, // A descriptive name for the final target node based on the user's goal (e.g., "Calculate Special Order Status"). Null if success is false.
  "required_configs": [{{ "type": "api"|"info", "id": number|string }}], // List containing ALL unique configurations (APIs and Infos) that are part of the dependency chain from start nodes to the target. Use the original 'id' from the configuration data. Null if success is false.
  "dependencies": {{ // Dictionary representing the directed graph edges (Parent -> Child). Key is the CHILD's prefixed ID, value is the PARENT's prefixed ID or a list of PARENT prefixed IDs for multiple dependencies. Include dependencies for the virtual target node 'target_0'. Null if success is false.
    "child_node_id_prefixed": "parent_node_id_prefixed",
    "another_child_id_prefixed": ["parent1_id_prefixed", "parent2_id_prefixed"],
    "target_0": "final_dependency_id_prefixed_or_list"
  }}
}}
```
Important Instructions for the JSON output:

Use prefixed IDs ONLY in start_node_ids and the keys/values within the dependencies dictionary (e.g., "api_1", "info_2").
The virtual target node representing the final goal MUST always be identified as "target_0" in the dependencies dictionary.
The required_configs list should contain objects with "type" and the original "id" (not prefixed). Include all nodes in the successful path.
If the target cannot be achieved (success: false), provide the reasoning, and set start_node_ids, target_node_name, required_configs, and dependencies to null.
Ensure the dependency flow is logical, connecting outputs to required inputs or inferred needs. """
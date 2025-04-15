import os
import json
import graphviz
import logging
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# --- Configure Google Generative AI ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables.")
    # You might want to exit or handle this more gracefully
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logging.info("Google Generative AI configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Google Generative AI: {e}")
        # Handle configuration error

# --- System Built-in Data ---
# (Keep SYSTEM_APIS and SYSTEM_INFOS as defined previously)
SYSTEM_APIS = [
    {"id": 1, "name": "订单详情接口", "serviceName": "merchantService", "methodName": "queryOrder", "inputs": ["orderId"], "outputs": ["orderStatus", "orderAmount", "orderDetail"]},
    {"id": 2, "name": "店铺详情接口", "serviceName": "shopService", "methodName": "queryShop", "inputs": ["shopId"], "outputs": ["shopName", "shopInfo"]},
    {"id": 3, "name": "用户基础信息接口", "serviceName": "userService", "methodName": "queryUser", "inputs": ["userId"], "outputs": ["userName", "userLevel"]}
]

SYSTEM_INFOS = [
    {"id": 1, "name": "订单id", "key": "orderId", "type": "input", "outputs": ["orderId"]}, # 标记为输入类型
    {"id": 2, "name": "订单金额", "key": "orderFee", "dependencies": [{"type": "api", "id": 1}], "outputs": ["orderFee"]}, # 简化：假设它能计算出orderFee
    {"id": 3, "name": "用户id", "key": "userId", "type": "input", "outputs": ["userId"]}, # 标记为输入类型
    {"id": 4, "name": "店铺id", "key": "shopId", "type": "input", "outputs": ["shopId"]}, # 标记为输入类型
    {"id": 5, "name": "订单状态", "key": "orderStatus", "dependencies": [{"type": "api", "id": 1}], "outputs": ["orderStatus"]} # 简化：假设它能计算出orderStatus
]


# --- Helper function to find full config dict by type and id ---
def get_full_config(config_type, config_id, available_apis, available_infos):
    """Finds the full configuration dictionary based on type and id."""
    if config_type == 'api':
        return find_config_by_id(available_apis, config_id)
    elif config_type == 'info':
        return find_config_by_id(available_infos, config_id)
    return None

# --- Function to find config by ID (needed by get_full_config) ---
def find_config_by_id(config_list, config_id):
    """Helper to find config by ID."""
    for config in config_list:
        # Ensure comparison handles potential string/int mismatch if ID comes from LLM
        if str(config.get('id')) == str(config_id):
            return config
    return None

# --- LLM Interaction ---

def build_llm_prompt(target_description, available_apis, available_infos):
    """Constructs the prompt for the LLM."""
    # Use json.dumps for proper formatting within the prompt
    apis_json_string = json.dumps(available_apis, indent=2, ensure_ascii=False)
    infos_json_string = json.dumps(available_infos, indent=2, ensure_ascii=False)

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
{{apis_json_string}}
json

Infos:
```json
{{infos_json_string}}

User Target Description:
"{{target_description}}"

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
Important Instructions for the JSON output:

Use prefixed IDs ONLY in start_node_ids and the keys/values within the dependencies dictionary (e.g., "api_1", "info_2").
The virtual target node representing the final goal MUST always be identified as "target_0" in the dependencies dictionary.
The required_configs list should contain objects with "type" and the original "id" (not prefixed). Include all nodes in the successful path.
If the target cannot be achieved (success: false), provide the reasoning, and set start_node_ids, target_node_name, required_configs, and dependencies to null.
Ensure the dependency flow is logical, connecting outputs to required inputs or inferred needs. """
    return prompt


def call_llm_understanding(target_description, available_apis, available_infos):
    """Calls the LLM to analyze dependencies and returns the structured JSON response."""
    if not GOOGLE_API_KEY:
        return {"success": False, "message": "LLM API Key not configured."}

    prompt = build_llm_prompt(target_description, available_apis, available_infos)
    logging.info("----- Sending Prompt to LLM -----")
    # Log only a part of the prompt if it's too long
    # logging.info(prompt[:1000] + "...")
    logging.info("----- End of Prompt -----")


    try:
        # Select the model - use a newer, capable model
        # model = genai.GenerativeModel('gemini-pro')
        # Using 1.5 Flash for potentially faster responses, adjust if needed
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Set safety settings to be less restrictive for this task if needed, but be cautious
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        # ]

        # Force JSON output if the model supports it (Check Gemini documentation)
        # Note: As of early 2024, direct JSON mode might be limited.
        # The prompt strongly instructs JSON output, which is often sufficient.
        response = model.generate_content(
            prompt,
            # safety_settings=safety_settings,
            generation_config=genai.types.GenerationConfig(
                 # candidate_count=1, # Default is 1
                 # stop_sequences=['\n\n'], # Optional: Stop generation earlier
                 # max_output_tokens=2048, # Adjust as needed
                 temperature=0.1 # Lower temperature for more deterministic/factual output
            )
        )

        logging.info("----- Received Response from LLM -----")
        # Check for blocked response due to safety
        if not response.candidates:
             logging.error("LLM response blocked or empty.")
             # Try to get blocking reason if available
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
             return {"success": False, "message": f"LLM response was blocked or empty. Reason: {block_reason}"}

        raw_response_text = response.text
        logging.info(raw_response_text)
        logging.info("----- End of LLM Response -----")

        # Clean the response: LLMs sometimes add markdown ```json ... ```
        if raw_response_text.strip().startswith("```json"):
            raw_response_text = raw_response_text.strip()[7:-3].strip()
        elif raw_response_text.strip().startswith("```"):
             raw_response_text = raw_response_text.strip()[3:-3].strip()

        # Parse the JSON response
        try:
            parsed_response = json.loads(raw_response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode LLM JSON response: {e}")
            logging.error(f"Raw response was: {raw_response_text}")
            return {"success": False, "message": "LLM response was not valid JSON."}

        # Basic validation of the parsed structure (can be made more robust)
        if not isinstance(parsed_response, dict) or 'success' not in parsed_response:
            logging.error("LLM JSON response is missing required 'success' field.")
            return {"success": False, "message": "LLM response JSON structure is invalid."}

        # If successful, validate other required fields based on the prompt's spec
        if parsed_response['success']:
            required_keys = ['reasoning', 'start_node_ids', 'target_node_name', 'required_configs', 'dependencies']
            if not all(key in parsed_response for key in required_keys):
                logging.error(f"Successful LLM response is missing one or more required fields: {required_keys}")
                return {"success": False, "message": "LLM response JSON structure is incomplete for a successful result."}
            # Add more type checks if necessary (e.g., check if lists/dicts are correct type)

        return parsed_response

    except Exception as e:
        logging.error(f"Error calling Google Generative AI: {e}", exc_info=True)
        return {"success": False, "message": f"An error occurred during the LLM API call: {e}"}

# Graph Generation (Keep generate_dependency_graph as previously defined)
def generate_dependency_graph(full_required_configs_with_target, dependencies, target_node_name):
    """
    Generates the dependency graph using Graphviz.
    Expects full configuration dictionaries in full_required_configs_with_target.
    """
    dot = graphviz.Digraph(comment='Configuration Dependency Graph', format='svg')
    dot.attr(rankdir='LR') # Left-to-right layout

    node_render_ids = set() # Track nodes already added to the graph body

    # Add nodes
    for config_dict in full_required_configs_with_target:
        # Determine node properties based on the dictionary
        config_type = config_dict.get('type') # 'api', 'info', or 'target'
        config_id = config_dict.get('id')
        node_id = f"{config_type}_{config_id}" # Prefixed ID for graphviz

        if node_id in node_render_ids: # Avoid adding duplicate nodes
             continue

        label = config_dict.get('name', f"Unknown {config_type}")
        shape = 'box'
        color = 'lightblue' # Default for Info

        if config_type == 'api':
            shape = 'ellipse'
            color = 'lightcoral'
        elif config_type == 'info' and config_dict.get('is_input'): # Check if it's an input info
             shape = 'parallelogram'
             color = 'lightgrey'
        elif config_type == 'target':
             shape = ' Mrecord' # Maybe use a different shape like doublecircle? or keep Mrecord
             color = 'lightgreen'
             label = f"目标: {target_node_name}" # Use the name from LLM

        dot.node(node_id, label=label, shape=shape, style='filled', fillcolor=color)
        node_render_ids.add(node_id)

    # Add edges from dependencies
    if dependencies: # Ensure dependencies exist before iterating
        for child_id_str, parent_info in dependencies.items():
            parent_id_strs = parent_info if isinstance(parent_info, list) else [parent_info]

            # Ensure child node exists before adding edge
            if child_id_str not in node_render_ids:
                 logging.warning(f"Dependency specified for child node '{child_id_str}', but this node was not found in required_configs. Skipping edge.")
                 continue

            for parent_id_str in parent_id_strs:
                # Ensure parent node exists before adding edge
                if parent_id_str not in node_render_ids:
                     logging.warning(f"Dependency '{parent_id_str}' -> '{child_id_str}' specified, but parent node '{parent_id_str}' not found in required_configs. Skipping edge.")
                     continue
                dot.edge(parent_id_str, child_id_str)

    try:
        svg_data = dot.pipe(format='svg').decode('utf-8')
        return svg_data
    except graphviz.backend.execute.ExecutableNotFound:
        logging.error("Graphviz executable not found. Ensure Graphviz is installed and in PATH.")
        return None # Indicate graph generation failure
    except Exception as e:
        logging.error(f"Error generating Graphviz graph: {e}", exc_info=True)
        return None # Indicate graph generation failure

# Flask Routes
@app.route('/', methods=['GET'])
def index():
    """Display the main form."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_request():
    """Handle user input, call LLM, generate graph, and return result."""
    target_description = request.form.get('target_description', '')
    extra_info_str = request.form.get('extra_info', '')

    if not target_description:
        return render_template('index.html', error="目标配置描述不能为空。")

    # Combine system configs and extra info
    current_apis = list(SYSTEM_APIS)
    current_infos = list(SYSTEM_INFOS)
    extra_apis_parsed = []

    if extra_info_str:
        try:
            extra_data = json.loads(extra_info_str)
            if isinstance(extra_data, list):
                valid_extra_apis = [api for api in extra_data if isinstance(api, dict) and 'id' in api and 'name' in api]
                current_apis.extend(valid_extra_apis)
                extra_apis_parsed = valid_extra_apis
                logging.info(f"Added {len(valid_extra_apis)} extra API configurations.")
            else:
                 logging.warning("Extra info provided but is not a JSON list.")
                 # Optionally return an error message here
        except json.JSONDecodeError:
             return render_template('index.html',
                                    target_description=target_description,
                                    extra_info=extra_info_str,
                                    error="额外信息不是有效的JSON格式。")

    # --- Call the actual LLM ---
    llm_result = call_llm_understanding(target_description, current_apis, current_infos)

    # --- Process LLM Result ---
    if not llm_result or not isinstance(llm_result, dict):
         return render_template('index.html',
                                target_description=target_description,
                                extra_info=extra_info_str,
                                error="LLM调用失败或返回格式错误。")

    if llm_result.get("success"):
        # --- Prepare data for graph generation ---
        required_config_refs = llm_result.get("required_configs", [])
        dependencies = llm_result.get("dependencies", {})
        target_node_name = llm_result.get("target_node_name", "Unnamed Target")
        start_node_prefixed_ids = llm_result.get("start_node_ids", []) # Get prefixed start node IDs

        # 1. Map required config references back to full dictionaries
        full_required_configs = []
        if required_config_refs:
            for ref in required_config_refs:
                config_dict = get_full_config(ref.get('type'), ref.get('id'), current_apis, current_infos)
                if config_dict:
                     # Mark input infos based on LLM result for graph styling
                     prefixed_id = f"{ref.get('type')}_{ref.get('id')}"
                     if prefixed_id in start_node_prefixed_ids:
                          config_dict['is_input'] = True # Add a flag for graph generator
                     else:
                          config_dict['is_input'] = False
                     full_required_configs.append(config_dict)
                else:
                     logging.warning(f"LLM required config {ref} not found in available configs.")

        # 2. Add the virtual target node information for the graph
        target_node_dict = {
            "id": "0", # Consistent with "target_0" prefix
            "name": target_node_name,
            "type": "target" # Special type for graph generator
        }
        full_required_configs_with_target = full_required_configs + [target_node_dict]


        # --- Generate Graph ---
        graph_svg = generate_dependency_graph(
            full_required_configs_with_target,
            dependencies,
            target_node_name
        )

        if graph_svg:
            # Prepare display data (excluding the virtual target node for the list)
            used_configs_display = [cfg for cfg in full_required_configs if cfg.get('type') != 'target']
            # Convert start node prefixed IDs back to names for display
            start_node_names = []
            for prefixed_id in start_node_prefixed_ids:
                parts = prefixed_id.split('_', 1)
                if len(parts) == 2:
                    start_node_dict = get_full_config(parts[0], parts[1], current_apis, current_infos)
                    if start_node_dict:
                        start_node_names.append(start_node_dict.get('name', prefixed_id))
                    else:
                         start_node_names.append(prefixed_id) # Fallback to ID if not found

            return render_template('index.html',
                                   target_description=target_description,
                                   extra_info=extra_info_str,
                                   graph_svg=graph_svg,
                                   llm_reasoning=llm_result.get("reasoning", "No reasoning provided."),
                                   start_nodes=start_node_names, # Display names
                                   target_node=target_node_name,
                                   used_configs=used_configs_display # List of full dicts
                                   )
        else:
            # Graph generation failed even if LLM succeeded
             return render_template('index.html',
                                target_description=target_description,
                                extra_info=extra_info_str,
                                error="LLM分析成功，但生成流程图失败。请检查Graphviz安装和日志。",
                                llm_reasoning=llm_result.get("reasoning"))
    else:
        # LLM analysis failed (success: false)
        return render_template('index.html',
                                target_description=target_description,
                                extra_info=extra_info_str,
                                error=f"LLM未能规划出有效的配置流程: {llm_result.get('reasoning', llm_result.get('message', '未知错误'))}")

if __name__ == "__main__":
    # Ensure Graphviz is in PATH or configure as needed
    # os.environ["PATH"] += os.pathsep + '/path/to/graphviz/bin'
    app.run(debug=True, host='0.0.0.0', port=5001) # Use a different port if 5000 is busy, allow external access if needed
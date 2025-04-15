import os
import json
from turtle import config_dict

import graphviz
import logging
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
from dotenv import load_dotenv
from modelCalls import model_call

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
        config_dict = find_config_by_id(available_apis, config_id)
        config_dict.update({'type': 'api'})
        return config_dict
    elif config_type == 'info':
        config_dict = find_config_by_id(available_infos, config_id)
        config_dict.update({'type': 'info'})
        return config_dict
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
    你是一个用于分析具有API和信息节点的数据处理系统的配置依赖关系的专家。

    ### 系统描述：
    - API：代表外部接口。接收特定'输入'并产生'输出'
    - INFO：代表起始数据点或数据处理/计算逻辑
        - 类型为'input'的信息节点是流程的必要起点，通过其'outputs'提供初始数据
        - 其他信息节点可能依赖API或其他信息节点，需要依赖项的数据来执行逻辑（通过'name'或系统目标隐含），并通过'outputs'产生数据

    ### 可用配置：
    API列表：
    ```json
    {apis_json_string}
    ```

    INFO列表：
    ```json
    {infos_json_string}
    ```

    用户目标描述：
    "{target_description}"

    ### 分析任务：
    根据目标描述识别必需的'input'型起始信息节点
    确定从输入到最终目标所需处理的API和信息节点序列，匹配各步骤的'outputs'与后续步骤的'inputs'需求
    定义代表最终目标的"target_0"虚拟节点
    映射所有必需配置间的依赖关系

    ### 输出格式：
    仅返回单个合法JSON对象，严格遵循以下结构：

    ```json
    {{
    "success": boolean, // 目标是否可达
      "reasoning": string, // 成功时的依赖路径说明/失败原因
      "start_node_ids": [string], // 必需的起始信息节点ID列表（如["info_1"]）
      "target_node_name": string, // 目标节点描述性名称（如"计算特殊订单状态"）
      "required_configs": [{{ //列表包含所有唯一配置（API和INFO），这些配置是从开始节点到目标的依赖关系链的一部分。使用配置数据中的原始“ ID”。如果目标不可达则为null。
        "type": "api|info",
        "id": number|string // 原始ID
      }}],
      "dependencies": {{ // 代表有向图边缘的字典（parent-> child）。key是child的带前缀ID，value是parent的带前缀ID，如果child依赖多个parent,value是parent的带前缀列表。包括虚拟目标节点“target_0”的依赖项，如果目标不可达则为null.
        "child_node_id_prefixed": "parent_node_id_prefixed",
        "another_child_id_prefixed": ["parent1_id_prefixed", "parent2_id_prefixed"],
        "target_0": "final_dependency_id_prefixed_or_list"
      }}
    }}

    ### 重要规范：
    依赖关系键使用带前缀ID（如"api_1"）
    必须包含"target_0"虚拟节点
    required_configs使用原始ID
    失败时相关字段设为null
    确保依赖流逻辑正确，输出到输入正确衔接
    """
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
        raw_response_text = model_call.call_deepseek_V3(prompt)
        logging.info(f"call model finish, res is {raw_response_text}")
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
            logging.info(f"node_render_ids is {node_render_ids}")
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
    target_description = request.form.get('goal', '')
    extra_info_str = request.form.get('extra_api_info', '')

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
            try:
                used_configs_json_string = json.dumps(used_configs_display, indent=2, ensure_ascii=False)
            except TypeError as e:
                logging.error(f"Error converting used_configs to JSON: {e}")
                used_configs_json_string = "Error formatting configuration data."
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
                                   used_configs=used_configs_json_string # List of full dicts
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
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
      "required_configs": [{{ // 依赖链中的所有配置
        "type": "api|info",
        "id": "number|string" // 原始ID
      }}],
      "dependencies": {{ // 依赖关系图（父节点->子节点）
        "子节点带前缀ID": "父节点带前缀ID",
        "target_0": "最终依赖项ID"
      }}
    }}

    ### 重要规范：
    依赖关系键使用带前缀ID（如"api_1"）
    必须包含"target_0"虚拟节点
    required_configs使用原始ID
    失败时相关字段设为null
    确保依赖流逻辑正确，输出到输入正确衔接
    """

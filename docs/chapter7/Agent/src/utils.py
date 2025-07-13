import inspect
from datetime import datetime
import pprint

def function_to_json(func) -> dict:
    # Define the mapping of Python types to JSON data types
    type_map = {
        str: "string",       # String type mapped to "string" of JSON
        int: "integer",      # Integer type mapped to JSON "integer"
        float: "number",     # "number" with floating point mapped to JSON
        bool: "boolean",     # Boolean mapped to JSON "boolean"
        list: "array",       # List type mapped to "array" of JSON
        dict: "object",      # "object" with dictionary type mapped to JSON
        type(None): "null",  # None type mapped to "null" of JSON
    }

    # Get the signature information of the function
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # If the signature is failed, an exception is thrown and a specific error message is displayed.
        raise ValueError(
            f"Unable to get the signature of function {func.__name__}: {str(e)}"
        )

    # Dictionary for storing parameter information
    parameters = {}
    for param in signature.parameters.values():
        # Try to get the type of the parameter. If the corresponding type cannot be found, the default setting is "string"
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # If the parameter type is not in type_map, an exception is thrown and a specific error message is displayed.
            raise KeyError(
                f"Unknown type annotation {param.annotation}, parameter name {param.name}: {str(e)}"
            )
        # Add parameter name and its type information to the parameter dictionary
        parameters[param.name] = {"type": param_type}

    # Get all required parameters in the function (i.e. parameters without default values)
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # Returns a dictionary containing function description information
    return {
        "type": "function",
        "function": {
            "name": func.__name__,            # The name of the function
            "description": func.__doc__ or "", # The document string of the function (empty string if it does not exist)
            "parameters": {
                "type": "object",
                "properties": parameters,     # Type description of function parameters
                "required": required,         # List of required parameters
            },
        },
    }


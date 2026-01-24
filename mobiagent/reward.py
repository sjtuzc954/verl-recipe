import json
from typing import Any

def calculate_iou(bbox1, bbox2):
    """Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0.0 and 1.0
    """
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Calculate intersection area
    if x1_inter < x2_inter and y1_inter < y2_inter:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        intersection = 0
    
    # Calculate union area
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    # if 0.0 < iou <= 0.3:
    #     return 0.0
    # elif 0.3 < iou <= 0.7:
    #     return 0.5
    # else:
    #     return 1.0
    return iou

def is_center_inside(solution_bbox, ground_truth_bbox):
    """Check if the center point of solution_bbox is inside ground_truth_bbox.
    
    Args:
        solution_bbox: [x1, y1, x2, y2]
        ground_truth_bbox: [x1, y1, x2, y2]
    
    Returns:
        1.0 if center is inside, 0.0 otherwise
    """
    # Calculate center of solution_bbox
    center_x = (solution_bbox[0] + solution_bbox[2]) / 2
    center_y = (solution_bbox[1] + solution_bbox[3]) / 2
    
    # Check if center is inside ground_truth_bbox
    if (ground_truth_bbox[0] <= center_x <= ground_truth_bbox[2] and
        ground_truth_bbox[1] <= center_y <= ground_truth_bbox[3]):
        return 1.0
    else:
        return 0.0

# from GUI-G1
def calculate_r_box(solution_bbox, ground_truth_bbox):
    x1, y1, x2, y2 = ground_truth_bbox
    x_hat1, y_hat1, x_hat2, y_hat2 = solution_bbox
    epsilon = 1e-6
    xp1 = 1 / (1 - abs(x1 - x_hat1) / 1000 + epsilon)
    xp2 = 1 / (1 - abs(x2 - x_hat2) / 1000 + epsilon)
    yp1 = 1 / (1 - abs(y1 - y_hat1) / 1000 + epsilon)
    yp2 = 1 / (1 - abs(y2 - y_hat2) / 1000 + epsilon)
    return 4 / (xp1 + xp2 + yp1 + yp2)

def calculate_bbox_score(solution_bbox, ground_truth_bbox):
    """Calculate bbox score combining IoU and center point check.
    
    Args:
        solution_bbox: [x1, y1, x2, y2]
        ground_truth_bbox: [x1, y1, x2, y2]
    
    Returns:
        Score between 0.0 and 1.0 (0.5 * IoU + 0.5 * center_inside)
    """
    # Part 1: IoU score
    iou_score = calculate_iou(solution_bbox, ground_truth_bbox)
    
    # Part 2: Center point inside check
    center_score = is_center_inside(solution_bbox, ground_truth_bbox)

    # Part 3: R_box
    r_box = calculate_r_box(solution_bbox, ground_truth_bbox)
    
    # Total bbox score
    # return 0.5 * iou_score + 0.5 * center_score
    return (center_score + 0.25 * iou_score + 0.125 * r_box) / 1.375

def validate_action(action: dict[str, Any]):
    # only three keys in action
    if len(action) != 3:
        raise ValueError(f"Action must have exactly 3 keys, got {len(action)} keys")
    reasoning = action["reasoning"]
    action_type = action["action"]
    parameters = action["parameters"]
    if not isinstance(reasoning, str):
        raise ValueError(f"'reasoning' must be a string, got {type(reasoning).__name__}")
    if not isinstance(action_type, str):
        raise ValueError(f"'action' must be a string, got {type(action_type).__name__}")
    if not isinstance(parameters, dict):
        raise ValueError(f"'parameters' must be a dict, got {type(parameters).__name__}")

    if action_type == "click":
        if len(parameters) != 2:
            raise ValueError(f"'click' action requires 2 parameters, got {len(parameters)}")
        target_element = parameters["target_element"]
        if not isinstance(target_element, str):
            raise ValueError(f"'target_element' must be a string, got {type(target_element).__name__}")
        bbox = parameters["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"'bbox' must be a list of 4 elements, got {type(bbox).__name__} with length {len(bbox) if isinstance(bbox, list) else 'N/A'}")
        if not all(isinstance(x, int) for x in bbox):
            raise ValueError(f"All bbox elements must be integers, got {[type(x).__name__ for x in bbox]}")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 1000 or bbox[3] > 1000:
            raise ValueError(f"bbox coordinates must be in range [0, 1000], got {bbox}")
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError(f"bbox invalid: x1 must be <= x2 and y1 must be <= y2, got {bbox}")
    elif action_type == "input":
        if len(parameters) != 1:
            raise ValueError(f"'input' action requires 1 parameter, got {len(parameters)}")
        text = parameters["text"]
        if not isinstance(text, str):
            raise ValueError(f"'text' must be a string, got {type(text).__name__}")
    elif action_type == "swipe":
        if len(parameters) != 3:
            raise ValueError(f"'swipe' action requires 3 parameters, got {len(parameters)}")
        direction = parameters["direction"]
        if not isinstance(direction, str) or direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            raise ValueError(f"'direction' must be one of ['UP', 'DOWN', 'LEFT', 'RIGHT'], got {direction}")
        start_coords = parameters["start_coords"]
        if not isinstance(start_coords, list) or len(start_coords) != 2:
            raise ValueError(f"'start_coords' must be a list of 2 elements, got {type(start_coords).__name__} with length {len(start_coords) if isinstance(start_coords, list) else 'N/A'}")
        if not all(isinstance(x, int) for x in start_coords):
            raise ValueError(f"All start_coords elements must be integers, got {[type(x).__name__ for x in start_coords]}")
        end_coords = parameters["end_coords"]
        if not isinstance(end_coords, list) or len(end_coords) != 2:
            raise ValueError(f"'end_coords' must be a list of 2 elements, got {type(end_coords).__name__} with length {len(end_coords) if isinstance(end_coords, list) else 'N/A'}")
        if not all(isinstance(x, int) for x in end_coords):
            raise ValueError(f"All end_coords elements must be integers, got {[type(x).__name__ for x in end_coords]}")
    elif action_type == "click_input":
        if len(parameters) != 3:
            raise ValueError(f"'click_input' action requires 3 parameters, got {len(parameters)}")
        target_element = parameters["target_element"]
        if not isinstance(target_element, str):
            raise ValueError(f"'target_element' must be a string, got {type(target_element).__name__}")
        text = parameters["text"]
        if not isinstance(text, str):
            raise ValueError(f"'text' must be a string, got {type(text).__name__}")
        bbox = parameters["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"'bbox' must be a list of 4 elements, got {type(bbox).__name__} with length {len(bbox) if isinstance(bbox, list) else 'N/A'}")
        if not all(isinstance(x, int) for x in bbox):
            raise ValueError(f"All bbox elements must be integers, got {[type(x).__name__ for x in bbox]}")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 1000 or bbox[3] > 1000:
            raise ValueError(f"bbox coordinates must be in range [0, 1000], got {bbox}")
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise ValueError(f"bbox invalid: x1 must be <= x2 and y1 must be <= y2, got {bbox}")
    elif action_type == "wait":
        if len(parameters) != 0:
            raise ValueError(f"'wait' action requires 0 parameters, got {len(parameters)}")
    elif action_type == "done":
        if len(parameters) != 1:
            raise ValueError(f"'done' action requires 1 parameter, got {len(parameters)}")
        status = parameters["status"]
        if not isinstance(status, str) or status not in ["success", "suspended", "failed"]:
            raise ValueError(f"'status' must be one of ['success', 'suspended', 'failed'], got {status}")
    else:
        raise ValueError(f"Unknown action type: {action_type}")

def _compute_score_impl(data_source, solution_str, ground_truth, extra_info=None):
    try:
        ground_truth = json.loads(ground_truth)
        action = json.loads(solution_str)
        validate_action(action)
        
        # Compare action types (ignore reasoning)
        action_type = action["action"]
        ground_truth_type = ground_truth["action"]
        
        if action_type != ground_truth_type:
            return 0.0
        
        # Extract parameters
        solution_params = action["parameters"]
        ground_truth_params = ground_truth["parameters"]
        
        # Calculate reward based on action type
        if action_type == "click":
            solution_bbox = solution_params["bbox"]
            ground_truth_bbox = ground_truth_params["bbox"]
            
            # Calculate bbox score
            bbox_score = calculate_bbox_score(solution_bbox, ground_truth_bbox)
            return bbox_score
        
        elif action_type == "click_input":
            solution_bbox = solution_params["bbox"]
            ground_truth_bbox = ground_truth_params["bbox"]
            solution_text = solution_params["text"]
            ground_truth_text = ground_truth_params["text"]
            
            # Part 1: bbox score (0.5 weight)
            bbox_score = calculate_bbox_score(solution_bbox, ground_truth_bbox)
            
            # Part 2: text score (0.5 weight)
            text_score = int(solution_text == ground_truth_text)
            
            # Total reward
            reward = 0.5 * bbox_score + 0.5 * text_score
            return reward
        
        elif action_type == "input":
            solution_text = solution_params["text"]
            ground_truth_text = ground_truth_params["text"]
            
            # Only compare text
            text_score = int(solution_text == ground_truth_text)
            return float(text_score)
        
        elif action_type == "swipe":
            solution_direction = solution_params["direction"]
            ground_truth_direction = ground_truth_params["direction"]
            
            # Only compare direction
            direction_score = int(solution_direction == ground_truth_direction)
            return float(direction_score)
        
        elif action_type == "wait":
            return 1.0
        
        elif action_type == "done":
            solution_status = solution_params["status"]
            ground_truth_status = ground_truth_params["status"]
            
            # Compare status
            status_score = int(solution_status == ground_truth_status)
            return float(status_score)

        return 0.0
        
    except json.JSONDecodeError as e:
        print(f"[Reward] JSON decode error: {e}")
        return -1.0
    except (ValueError, KeyError) as e:
        print(f"[Reward] Validation error: {e}")
        return -1.0
    except Exception as e:
        print(f"[Reward] Unexpected error: {type(e).__name__}: {e}")
        return -1.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    score = _compute_score_impl(data_source, solution_str, ground_truth, extra_info)
    score = min(1.0, score)
    score = max(-1.0, score)
    acc = score
    return {
        "score": score,
        "acc": acc
    }
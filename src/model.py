import timm
import config

def create_model(model_name=None, num_classes=None, pretrained=True):
    model_name = model_name or config.MODEL_NAME
    num_classes = num_classes or config.NUM_CLASSES

    model = timm.create_model(model_name, 
    num_classes=num_classes, 
    pretrained=pretrained,
    drop_rate=config.DROP_RATE
    )

    return model

def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6
    }
    
import yaml
from omegaconf import DictConfig, OmegaConf

from .log import get_cur_time, logger, rich_handler

UNIMPORTANT_CFG = DictConfig(
    {
        "fields": ["gpus", "debug", "wandb", "env", "uid", "file_prefix"],
        "prefix": ["_"],
        "postfix": ["_path", "_file", "_dir"],
    }
)


def _filter_nested_keys(data, keep_key):
    if isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            if not keep_key(key):
                continue
            filtered[key] = _filter_nested_keys(value, keep_key)
        return filtered
    if isinstance(data, list):
        return [_filter_nested_keys(item, keep_key) for item in data]
    return data


def get_important_cfg(cfg: DictConfig, reserve_file_cfg: bool = True, unimportant_cfg=UNIMPORTANT_CFG):
    uimp_cfg = cfg.get("_unimportant_cfg", unimportant_cfg)
    cfg_obj = OmegaConf.to_object(cfg)

    def is_preserve(key: str):
        keep_file_cfg = key == "_file_" and reserve_file_cfg
        prefix_allowed = (not any(key.startswith(prefix) for prefix in uimp_cfg.prefix)) or keep_file_cfg
        postfix_allowed = not any(key.endswith(postfix) for postfix in uimp_cfg.postfix)
        field_allowed = key not in uimp_cfg.fields
        return prefix_allowed and postfix_allowed and field_allowed

    return _filter_nested_keys(cfg_obj, is_preserve)


def print_important_cfg(cfg, log_func=logger.info, use_rich_console: bool = True):
    imp_cfg = get_important_cfg(cfg, reserve_file_cfg=False)
    if OmegaConf.is_config(imp_cfg):
        imp_cfg = OmegaConf.to_container(imp_cfg, resolve=True)

    class _FlowSeqDumper(yaml.SafeDumper):
        pass

    def _represent_list(self, data):
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _FlowSeqDumper.add_representer(list, _represent_list)
    _FlowSeqDumper.add_representer(tuple, _represent_list)

    yaml_str = yaml.dump(
        imp_cfg,
        Dumper=_FlowSeqDumper,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()

    sweep_name = None
    try:
        sweep_name = cfg.get("sweep_name", None)
    except Exception:
        if isinstance(cfg, dict):
            sweep_name = cfg.get("sweep_name", None)

    rule_text = f"{sweep_name} {get_cur_time()}" if sweep_name else get_cur_time()

    console = rich_handler.console if use_rich_console else None
    if console is not None:
        console.rule(rule_text)
        console.print(yaml_str, highlight=False, markup=False)
        console.rule(rule_text)
    else:
        output = f"{rule_text}\n{yaml_str}"
        if log_func is None:
            print(output)
        else:
            log_func(output)


def _eval(*args, **kwargs):
    return eval(*args, **kwargs)


def div_up(x, y):
    return (x + y - 1) // y


def replace_dot_by_underscore(input_str: str):
    return input_str.replace(".", "_")


def rename_alias(alias: str):
    replace_dict = {"true": "T", "false": "F", "True": "T", "False": "F"}
    for key, value in replace_dict.items():
        alias = alias.replace(key, value)
    return alias


def register_omega_conf_resolver():
    OmegaConf.register_new_resolver("eval", _eval)
    OmegaConf.register_new_resolver("min", lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver("div_up", div_up)
    OmegaConf.register_new_resolver("replace_dot_by_underscore", replace_dot_by_underscore)
    OmegaConf.register_new_resolver("rename_alias", rename_alias)
    OmegaConf.register_new_resolver("to_str", lambda x: str(x))


def save_config(cfg: DictConfig, path, as_global: bool = True):
    OmegaConf.save(config=DictConfig(cfg), f=path)
    if as_global:
        with open(path, "r") as input_file:
            original_content = input_file.read()
        with open(path, "w") as output_file:
            output_file.write("# @package _global_\n" + original_content)
    return cfg

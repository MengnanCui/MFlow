# -*- coding: utf-8 -*-
"""
System-specific tools: `mflow tool <name> ...`. Each is a module with main(argv) -> int.
Add one by dropping a module here and registering it below.
"""

TOOLS = {
    "izo-formation": ("mflow.extras.izo_formation", "In-Zn-O formation energy vs In ratio from a MACE-evaluated xyz"),
}


def cmd_tool(args) -> int:
    import importlib
    from ..log import LOG, table
    if not args.name:
        LOG(table("mflow tool <name> ...", ["tool", "what it does"], [(k, v[1]) for k, v in TOOLS.items()],
                  styles=["key"]))
        return 0
    if args.name not in TOOLS:
        LOG(f"[err]unknown tool {args.name!r}[/err] — available: {', '.join(TOOLS)}")
        return 2
    return importlib.import_module(TOOLS[args.name][0]).main(args.rest)

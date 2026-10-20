# Marks `utils` as a regular package. Without this, `import utils.selfieseg_lib` under
# pytest resolves to ComfyUI's own top-level `utils` package (the portable Python puts
# the ComfyUI root on sys.path), because a regular package anywhere on sys.path beats
# a namespace directory. Inside ComfyUI the node uses relative imports, unaffected.

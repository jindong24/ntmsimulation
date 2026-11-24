"""Run the original mesh block; reject any mismatch with archived nodes."""

import argparse
import ast
from hashlib import sha256
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


def original_mesh_block(source):
    tree = ast.parse(source.read_bytes(), filename=str(source))

    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        )

    start, = [i for i, node in enumerate(tree.body) if assigns(node, "N")]
    end, = [i for i, node in enumerate(tree.body) if assigns(node, "TISSUE_mesh")]
    if not start < end:
        raise ValueError("Cannot identify the original mesh block")
    return ast.Module(body=tree.body[start:end + 1], type_ignores=[])


def verify_region(legacy, candidate, label, atol=1e-12):
    if legacy.shape != candidate.shape:
        raise ValueError(f"{label}: expected {len(legacy)} nodes, found {len(candidate)}")
    distance = np.max(np.abs(legacy[:, None, :] - candidate[None, :, :]), axis=-1)
    nearest = distance.argmin(axis=1)
    error = float(distance[np.arange(len(legacy)), nearest].max())
    if error > atol or len(set(nearest)) != len(legacy):
        raise ValueError(f"{label}: original coordinates not reproduced; max nearest error {error:.4g}")
    return nearest, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("coordinates", type=Path)
    parser.add_argument("--save-mesh", type=Path)
    parser.add_argument("--archive-connectivity", type=Path)
    args = parser.parse_args()
    code = compile(original_mesh_block(args.source), str(args.source), "exec")
    namespace = {"__name__": "original_mesh_check"}
    try:
        exec("from dolfin import *\nfrom mshr import *", namespace)
    except ImportError as error:
        parser.exit(2, f"Missing original mesh dependency: {error}\n")
    exec(code, namespace)
    mesh = namespace["mesh"]
    print(f"Original script: {args.source}, N={namespace['N']}")
    print(f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} triangles")
    print(f"Connectivity checksum: {sha256(np.asarray(mesh.cells()).tobytes()).hexdigest()}")
    reference = None
    if args.archive_connectivity:
        with np.load(args.archive_connectivity, allow_pickle=False) as archive:
            reference = {label: archive[label] for label in ("Mucus", "Tissue")}
    book = load_workbook(args.coordinates, read_only=True, data_only=True)
    try:
        for label, key in (("Mucus", "MUCUS_mesh"), ("Tissue", "TISSUE_mesh")):
            archived = np.array(list(book[label].values)[1:], dtype=float)
            region = namespace[key]
            order, error = verify_region(archived, region.coordinates(), label)
            print(f"{label}: matched {len(archived)} nodes (max error {error:.4g})")
            if reference is not None:
                candidate = {tuple(sorted(row)) for row in np.argsort(order)[region.cells()]}
                reconstructed = {tuple(sorted(row)) for row in reference[label]}
                print(f"{label}: {len(candidate & reconstructed)}/{len(candidate)} triangles "
                      "match the reconstructed archive connectivity")
    finally:
        book.close()
    if args.save_mesh:
        if args.save_mesh.exists():
            raise FileExistsError(args.save_mesh)
        writer = namespace["HDF5File"](mesh.mpi_comm(), str(args.save_mesh), "w")
        writer.write(mesh, "/mesh")
        writer.write(namespace["cell_mark"], "/cell_mark")
        writer.close()
    print("Coordinates match; triangle connectivity still needs independent archive verification.")


if __name__ == "__main__":
    main()

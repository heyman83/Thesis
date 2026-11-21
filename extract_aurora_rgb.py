"""Copy RGB JPGs from nested Aurora930_* captures into one flat folder."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.digest()


def find_images(source, output):
    for image in sorted(source.rglob('*')):
        if not image.is_file() or image.suffix.lower() not in {'.jpg', '.jpeg'}:
            continue
        resolved = image.resolve()
        if not resolved.is_relative_to(source) or resolved.is_relative_to(output):
            continue
        parents = []
        for parent in image.parents:
            parents.append(parent)
            if parent == source:
                break
        rgb = next((p for p in parents if p.name.lower() == 'rgb'), None)
        if rgb is None:
            continue
        scene = next((p for p in rgb.parents if p.name.lower().startswith('aurora930_')
                      and (p == source or p.is_relative_to(source))), None)
        if scene is not None:
            yield scene, image


def extract(source, output, dry_run=False):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f'Source folder does not exist: {source}')
    if source == output or source.is_relative_to(output):
        raise ValueError('Choose a separate output folder, not the source or a parent of it.')
    images = list(find_images(source, output))
    if not images:
        print('No JPG/JPEG images found inside RGB folders under Aurora930_* captures.')
        return 0, 0
    if dry_run:
        print(f'Would copy {len(images)} RGB images to: {output}')
        return len(images), 0
    output.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for scene, image in images:
        # Stable names keep separate scenes with identical camera filenames distinct.
        scene_name = re.sub(r'[^A-Za-z0-9_-]+', '_', scene.name).strip('_')[:70]
        stem = re.sub(r'[^A-Za-z0-9_-]+', '_', image.stem).strip('_')[:40] or 'image'
        key = hashlib.sha256(image.relative_to(source).as_posix().encode()).hexdigest()[:16]
        destination = output / f'{scene_name}_{stem}_{key}{image.suffix.lower()}'
        if destination.exists():
            if destination.stat().st_size == image.stat().st_size and digest(destination) == digest(image):
                skipped += 1
                continue
            raise FileExistsError(f'Destination already contains a different file: {destination}')
        # Exclusive creation ensures existing files are never overwritten.
        created = False
        try:
            with image.open('rb') as original, destination.open('xb') as target:
                created = True
                shutil.copyfileobj(original, target)
            shutil.copystat(image, destination)
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        copied += 1
    print(f'Copied: {copied} | Already present (skipped): {skipped}')
    print(f'Output folder: {output}')
    print('Original images are unchanged. No path log was created.')
    return copied, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='USB folder containing Aurora930_* captures')
    parser.add_argument('--output', required=True, type=Path, help='Destination for the flattened RGB images')
    parser.add_argument('--dry-run', action='store_true', help='Count matching images without copying anything')
    args = parser.parse_args()
    try:
        extract(args.source, args.output, args.dry_run)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()

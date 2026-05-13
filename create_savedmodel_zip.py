import zipfile
import os

source_dir = r"c:\Users\saini\Desktop\ats\ats-ai-core\model\unified_model\saved_model"
output_zip = r"c:\Users\saini\Desktop\ats\ats-ai-core\model\unified_model\unified_keras_savedmodel.zip"

print(f"Creating ZIP archive...")
print(f"Source: {source_dir}")
print(f"Destination: {output_zip}")

with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.join('saved_model', os.path.relpath(file_path, source_dir))
            print(f"Adding: {arcname}")
            zipf.write(file_path, arcname)

zip_size_mb = os.path.getsize(output_zip) / (1024 * 1024)
print(f"\n✓ ZIP created successfully!")
print(f"✓ Location: {output_zip}")
print(f"✓ Size: {zip_size_mb:.2f} MB")

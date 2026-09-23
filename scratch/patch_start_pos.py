with open("scratch/update_v4_generator.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace start_pos lookup
code = code.replace('start_pos = text.find(sec5_target)', 'start_pos = text.find("    # SECTION 5:") - 5')

with open("scratch/update_v4_generator.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Fixed start_pos search in update_v4_generator.py")

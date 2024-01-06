import os
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading

def convert_mp3_files(input_folder, progress_var, counter_var, root_window, error_var):
    output_folder = os.path.join(os.path.dirname(input_folder), f"{input_folder}_Converted")

    total_files = sum(len(files) for _, _, files in os.walk(input_folder))
    current_file = 0
    conversion_errors = []

    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for file in files:
            if file.endswith(".mp3"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_subfolder, f"{file}")

                command = [
                    'ffmpeg',
                    '-i', input_path,
                    '-ar', '16000',  # Sample rate: 16kHz
                    '-ac', '1',      # Channels: Mono
                    '-b:a', '32k',   # Bit rate: 32 kbps
                    output_path
                ]

                try:
                    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Converted: {input_path} -> {output_path}")
                except subprocess.CalledProcessError as e:
                    error_message = f"Error converting {input_path}: {e.stderr.decode('utf-8').strip()}"
                    print(error_message)
                    conversion_errors.append(error_message)

                current_file += 1
                progress_value = (current_file / total_files) * 100
                progress_var.set(progress_value)
                counter_var.set(f"Completed: {current_file}/{total_files}")

    if conversion_errors:
        error_var.set("\n".join(conversion_errors))
        messagebox.showerror("Conversion Errors", "\n".join(conversion_errors))

def choose_input_folder():
    folder_path = filedialog.askdirectory()
    input_folder_entry.delete(0, tk.END)
    input_folder_entry.insert(0, folder_path)

def convert_button_clicked():
    input_folder = input_folder_entry.get()

    # Create a progress bar, a counter label, and an error label
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, orient=tk.HORIZONTAL, length=300, mode='determinate')
    progress_bar.grid(row=2, column=1, pady=10)

    counter_var = tk.StringVar()
    counter_label = tk.Label(root, textvariable=counter_var)
    counter_label.grid(row=3, column=1, pady=5)

    error_var = tk.StringVar()
    error_label = tk.Label(root, textvariable=error_var, wraplength=400, justify=tk.LEFT, fg='red')
    error_label.grid(row=4, column=1, pady=5)

    # Create a new thread for the conversion process
    conversion_thread = threading.Thread(target=convert_mp3_files, args=(input_folder, progress_var, counter_var, root, error_var))
    conversion_thread.start()

# GUI setup
root = tk.Tk()
root.title("MP3 Specification Converter")

# Center the window on the screen
window_width = 500
window_height = 220
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Input Folder
input_folder_label = tk.Label(root, text="Input Folder:")
input_folder_label.grid(row=0, column=0, sticky="e")

input_folder_entry = tk.Entry(root, width=50)
input_folder_entry.grid(row=0, column=1, padx=5)

choose_folder_button = tk.Button(root, text="Choose Folder", command=choose_input_folder)
choose_folder_button.grid(row=0, column=2, padx=5)

# Convert Button
convert_button = tk.Button(root, text="Convert", command=convert_button_clicked)
convert_button.grid(row=1, column=1, pady=10)

root.mainloop()

from nonn import NoNN

# Instantiate NoNN class with default configs
nonn = NoNN(config_file="configs/default.config")

# Create the teacher
nonn.create_teacher()

# Execute the mode specified for the teacher (default = "train_and_partition")
nonn.teacher.execute()

# Create the student (the unified student model is created automatically)
nonn.create_student()

# Train the student
nonn.student.train_model()

# Deploy student
nonn.student.deploy()

from src.exercises.manager import ExerciseManager


class SessionManager:

    AVAILABLE_EXERCISES = {
        "1": "squat",
        "2": "sit_up",
        "3": "push_up",
        "4": "all",
    }

    def select_exercise(self) -> str:
        print("\n=== SportMotion ===")
        print("Select exercise:")
        print("1. Squat")
        print("2. Sit-up")
        print("3. Push-up")
        print("4. All")

        while True:
            choice = input("Enter your choice: ").strip()

            if choice in self.AVAILABLE_EXERCISES:
                return self.AVAILABLE_EXERCISES[choice]

            print("Invalid choice. Please select 1-4.")

    def create_exercise_manager(self) -> ExerciseManager:
        exercise_name = self.select_exercise()

        if exercise_name != "squat":
            raise NotImplementedError(
                f"'{exercise_name}' is not implemented yet."
            )

        return ExerciseManager(
            exercise_name
        )
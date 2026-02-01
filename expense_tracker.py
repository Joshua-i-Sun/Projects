import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


FILE_NAME = "expenses.csv"


def load_expenses():
    """Load expenses from CSV file or create an empty DataFrame."""
    if os.path.exists(FILE_NAME):
        return pd.read_csv(FILE_NAME)
    else:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Note"])


def save_expenses(expenses: pd.DataFrame):
    """Save expenses to CSV file."""
    expenses.to_csv(FILE_NAME, index=False)


def add_expense(expenses: pd.DataFrame, category: str, amount: float, note: str):
    """Add a new expense entry."""
    new_expense = {
        "Date": datetime.today().strftime("%Y-%m-%d"),
        "Category": category,
        "Amount": amount,
        "Note": note,
    }
    expenses = pd.concat([expenses, pd.DataFrame([new_expense])], ignore_index=True)
    save_expenses(expenses)
    print("Expense added successfully!")
    return expenses


def view_summary(expenses: pd.DataFrame):
    """View summary of expenses by category."""
    if expenses.empty:
        print("No expenses recorded yet.")
        return
    summary = expenses.groupby("Category")["Amount"].sum()
    print("\n=== Expense Summary ===")
    print(summary)
    print("=======================")


def plot_expenses(expenses: pd.DataFrame):
    """Plot expenses by category as a pie chart."""
    if expenses.empty:
        print("No expenses to plot.")
        return
    summary = expenses.groupby("Category")["Amount"].sum()
    summary.plot(kind="pie", autopct="%1.1f%%", figsize=(6, 6))
    plt.title("Expenses by Category")
    plt.ylabel("")
    plt.show()


def main():
    expenses = load_expenses()

    while True:
        print("\n=== Personal Expense Tracker ===")
        print("1. Add Expense")
        print("2. View Summary")
        print("3. Plot Expenses")
        print("4. Exit")

        choice = input("Choose an option (1-4): ")

        if choice == "1":
            category = input("Enter category (e.g., Food, Transport, Entertainment): ")
            amount = float(input("Enter amount: "))
            note = input("Enter note (optional): ")
            expenses = add_expense(expenses, category, amount, note)

        elif choice == "2":
            view_summary(expenses)

        elif choice == "3":
            plot_expenses(expenses)

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()

"""
One-off CLI utility to generate a bcrypt hash for APP_PASSWORD_HASH.

Usage:
    python -m auth.generate_password_hash
"""
import getpass
import bcrypt


def main():
    pwd = getpass.getpass("Enter the password you want to hash: ")
    confirm = getpass.getpass("Confirm password: ")
    if pwd != confirm:
        print("Passwords do not match. Try again.")
        return
    if not pwd:
        print("Password cannot be empty.")
        return

    hashed = bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt())
    print("\nAdd this line to your .env file:\n")
    print(f"APP_PASSWORD_HASH={hashed.decode('utf-8')}")


if __name__ == "__main__":
    main()

# Contextual Adaptation Engine (CAE)
import platform

def detect_user_context():
    '''
    Detects user's device type, OS, and system settings.
    :return: Dictionary with system context information.
    '''
    return {
        "Device Type": "Desktop" if platform.system() in ["Windows", "Linux", "MacOS"] else "Mobile",
        "Operating System": platform.system(),
        "OS Version": platform.version(),
        "Python Version": platform.python_version(),
        "Processor": platform.processor(),
        "Architecture": platform.architecture()[0]
    }

def detect_ui_preferences():
    '''
    Simulated detection of user UI preferences.
    :return: Dictionary with UI settings.
    '''
    return {
        "Dark Mode Enabled": False,
        "High Contrast Mode": False,
        "Preferred Font Size": "Medium",
        "Touchscreen Available": "Unknown"
    }

def generate_ui_adaptation(context, preferences):
    '''
    Generates UI adaptation recommendations based on detected user context.
    :param context: User system information.
    :param preferences: UI preferences.
    :return: Dictionary with UI recommendations.
    '''
    return {
        "Theme": "Dark Mode" if preferences["Dark Mode Enabled"] else "Light Mode",
        "Contrast": "High Contrast Enabled" if preferences["High Contrast Mode"] else "Standard Contrast",
        "Layout": "Responsive Mobile Layout" if context["Device Type"] == "Mobile" else "Desktop Adaptive Layout",
        "Touch Optimization": "Enabled" if context["Device Type"] == "Mobile" else "Disabled",
        "Font Size": preferences["Preferred Font Size"]
    }

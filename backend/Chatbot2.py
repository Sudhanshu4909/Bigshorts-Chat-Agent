# Complete Bigshorts chatbot using local LLM with all original tools and functionality
from llama_cpp import Llama
import yaml
from typing import Dict, List, Union
import random
import re
import os

# Define strict allowed parameters
ALLOWED_CONTENT_TYPES = [
    "shot", "snip", "ssup", "collab",
    "editing a shot", "invite friends", "feedback", "multiple accounts", 
    "account overview", "store draft", "change password", "notification", 
    "change theme", "report", "moment", "delete post", "post insights", 
    "saved posts", "edit profile", "edit post", "block/unblock user", 
    "hide/unhide users", "messages", "discovery", "editing a ssup", 
    "interactive snip", "flix", "create a playlist", "editing a flix", 
    "editing a snip"
]
ALLOWED_ISSUE_TYPES = [
    "login", "upload", "notification", "privacy", 
    "account", "payment", "content", "technical", "app", 
    "video", "audio", "connection", "quality", "blocking", 
    "reporting", "messaging", "password", "theme"
]
ALLOWED_PLATFORM_SECTIONS = [
    "profile", "messages", "settings", "shot", "snip", "ssup", "collab",
    "discovery", "saved", "drafts", "notifications", "feedback", "moments", 
    "playlist", "flix", "account", "insights", "themes", "blocking", 
    "hiding", "reporting", "editing", "interactive"
]

# Content type mapping for standardization
CONTENT_TYPE_MAPPING = {
    # SHOT mappings
    "photo": "shot",
    "picture": "shot", 
    "image": "shot", 
    "pic": "shot", 
    "pics": "shot",
    "photograph": "shot", 
    "snapshot": "shot", 
    "photography": "shot", 
    "pictures": "shot", 
    "shot photo": "shot",
    "create shot": "shot",
    "make shot": "shot",
    "how to shot": "shot",
    "how to create shot": "shot",
    "how to make shot": "shot",
    "how to create a shot": "shot",
    "how to make a shot": "shot",
    
    # SNIP mappings (excluding interactive)
    "video": "snip", 
    "clip": "snip", 
    "reel": "snip", 
    "short": "snip", 
    "shorts": "snip",
    "reels": "snip", 
    "videos": "snip", 
    "short video": "snip", 
    "short-form video": "snip",
    "tiktok-style": "snip",
    "create snip": "snip",
    "make snip": "snip",
    "how to snip": "snip",
    "how to create snip": "snip",
    "how to make snip": "snip",
    "how to create a snip": "snip",
    "how to make a snip": "snip",
    
    # SSUP mappings
    "story": "ssup", 
    "stories": "ssup", 
    "temporary": "ssup", 
    "24 hour": "ssup", 
    "vanishing": "ssup",
    "disappearing": "ssup", 
    "temporary post": "ssup", 
    "daily update": "ssup", 
    "status update": "ssup",
    "instagram-style story": "ssup", 
    "status": "ssup",
    "create ssup": "ssup",
    "make ssup": "ssup",
    "how to ssup": "ssup",
    "how to create ssup": "ssup",
    "how to make ssup": "ssup",
    "how to create a ssup": "ssup",
    "how to make a ssup": "ssup",
    
    # COLLAB mappings
    "collaboration": "collab", 
    "together": "collab", 
    "partner": "collab", 
    "joint": "collab", 
    "group": "collab",
    "duo": "collab", 
    "team": "collab", 
    "cooperative": "collab", 
    "with friend": "collab", 
    "with someone": "collab",
    "create collab": "collab",
    "make collab": "collab",
    "how to collab": "collab",
    "how to create collab": "collab",
    "how to make collab": "collab",
    "how to create a collab": "collab",
    "how to make a collab": "collab",
    "how to collaborate": "collab",
    
    # FLIX mappings
    "create flix": "flix", 
    "flix video": "flix", 
    "long video": "flix", 
    "episode": "flix",
    "make flix": "flix",
    "how to flix": "flix",
    "how to create flix": "flix",
    "how to make flix": "flix",
    "how to create a flix": "flix",
    "how to make a flix": "flix",
    
    # EDITING A SHOT mappings
    "edit shot": "editing a shot", 
    "modify shot": "editing a shot",
    "editing shot": "editing a shot",
    "how to edit shot": "editing a shot",
    "how to edit a shot": "editing a shot",
    "how to modify shot": "editing a shot",
    "how to modify a shot": "editing a shot",
    
    # EDITING A SNIP mappings
    "edit snip": "editing a snip", 
    "modify snip": "editing a snip", 
    "change snip": "editing a snip",
    "editing snip": "editing a snip",
    "how to edit snip": "editing a snip",
    "how to edit a snip": "editing a snip",
    "how to modify snip": "editing a snip",
    "how to modify a snip": "editing a snip",
    "how to change snip": "editing a snip",
    "how to change a snip": "editing a snip",
    
    # EDITING A SSUP mappings
    "edit ssup": "editing a ssup", 
    "modify ssup": "editing a ssup", 
    "change ssup": "editing a ssup",
    "editing ssup": "editing a ssup",
    "how to edit ssup": "editing a ssup",
    "how to edit a ssup": "editing a ssup",
    "how to modify ssup": "editing a ssup",
    "how to modify a ssup": "editing a ssup",
    "how to change ssup": "editing a ssup",
    "how to change a ssup": "editing a ssup",
    
    # EDITING A FLIX mappings
    "edit flix": "editing a flix", 
    "modify flix": "editing a flix", 
    "change flix": "editing a flix",
    "editing flix": "editing a flix",
    "how to edit flix": "editing a flix",
    "how to edit a flix": "editing a flix",
    "how to modify flix": "editing a flix",
    "how to modify a flix": "editing a flix",
    "how to change flix": "editing a flix",
    "how to change a flix": "editing a flix",
    
    # INTERACTIVE SNIP mappings - separated from regular snip for clarity
    "interactive video": "interactive snip", 
    "interactive": "interactive snip", 
    "add button": "interactive snip", 
    "clickable": "interactive snip",
    "interactive snip video": "interactive snip",
    "create interactive": "interactive snip",
    "make interactive": "interactive snip",
    "create interactive snip": "interactive snip",
    "make interactive snip": "interactive snip",
    "how to interactive": "interactive snip",
    "how to interactive snip": "interactive snip",
    "how to create interactive": "interactive snip",
    "how to make interactive": "interactive snip",
    "how to create interactive snip": "interactive snip",
    "how to make interactive snip": "interactive snip",
    "how to create an interactive snip": "interactive snip",
    "how to make an interactive snip": "interactive snip",
    
    # INVITE FRIENDS mappings
    "invite friend": "invite friends", 
    "add friend": "invite friends", 
    "add friends": "invite friends",
    "how to invite friend": "invite friends",
    "how to invite friends": "invite friends",
    "how to add friend": "invite friends",
    "how to add friends": "invite friends",
    "How to invite friends": "invite friends",
    "How do i invite my friends": "invite friends",
    "How do i Share bigshorts": "invite friends",
    "How to invite friends on Bigshorts": "invite friends",
    
    # FEEDBACK mappings
    "give feedback": "feedback", 
    "submit feedback": "feedback", 
    "suggestion": "feedback",
    "how to give feedback": "feedback",
    "how to submit feedback": "feedback",
    "how to provide feedback": "feedback",
    
    # MULTIPLE ACCOUNTS mappings
    "multiple account": "multiple accounts", 
    "switch account": "multiple accounts", 
    "add account": "multiple accounts",
    "how to add account": "multiple accounts",
    "how to add multiple accounts": "multiple accounts",
    "how to switch account": "multiple accounts",
    "how to switch accounts": "multiple accounts",
    "how to manage multiple accounts": "multiple accounts",
    "multiple profile": "multiple accounts",
    "multiple profiles": "multiple accounts",
    "change profile": "multiple accounts",
    "switch profile": "multiple accounts",
    "Second account": "multiple accounts",
    
    # ACCOUNT OVERVIEW mappings
    "account stats": "account overview", 
    "overview": "account overview", 
    "analytics": "account overview",
    "how to view account stats": "account overview",
    "how to check analytics": "account overview",
    "how to see account overview": "account overview",
    
    # STORE DRAFT mappings
    "save draft": "store draft", 
    "draft": "store draft", 
    "save content": "store draft",
    "how to save draft": "store draft",
    "how to store draft": "store draft",
    "how to save content": "store draft",
    
    # CHANGE PASSWORD mappings
    "password": "change password", 
    "update password": "change password", 
    "new password": "change password",
    "how to change password": "change password",
    "how to update password": "change password",
    "how to reset password": "change password",
    
    # NOTIFICATION mappings
    "notifications": "notification", 
    "alerts": "notification", 
    "notice": "notification",
    "how to check notifications": "notification",
    "how to view notifications": "notification",
    "how to manage notifications": "notification",
    
    # CHANGE THEME mappings
    "theme": "change theme", 
    "dark mode": "change theme", 
    "light mode": "change theme", 
    "appearance": "change theme",
    "how to change theme": "change theme",
    "how to switch theme": "change theme",
    "how to change appearance": "change theme",
    "How to change app colour": "change theme",
    "App colour": "change theme",
    "App color": "change theme",
    
    # REPORT mappings
    "flag content": "report", 
    "report content": "report", 
    "report user": "report", 
    "abuse": "report",
    "how to report": "report",
    "how to flag content": "report",
    "how to report content": "report",
    "how to report a user": "report",
    
    # MOMENT mappings
    "create moment": "moment", 
    "moments": "moment", 
    "memory": "moment", 
    "memories": "moment",
    "highlight": "moment",
    "highlights": "moment",
    "story highlight": "moment",
    "story highlights": "moment",
    "ssup highlight": "moment",
    "ssup highlights": "moment",
    "save story": "moment",
    "save ssup": "moment",
    "save stories": "moment",
    "save ssups": "moment",
    "archived story": "moment",
    "archived stories": "moment",
    "archived ssup": "moment",
    "archived ssups": "moment",
    "permanent story": "moment",
    "permanent ssup": "moment",
    "how to create moment": "moment",
    "how to make moment": "moment",
    "how to create a moment": "moment",
    "how to make a moment": "moment",
    "how to save story": "moment",
    "how to save ssup": "moment",
    "how to create highlight": "moment",
    "how to make highlight": "moment",

    
    # DELETE POST mappings
    "remove post": "delete post", 
    "erase post": "delete post", 
    "delete content": "delete post",
    "how to delete post": "delete post",
    "how to remove post": "delete post",
    "how to delete a post": "delete post",
    "how to remove a post": "delete post",
    
    # POST INSIGHTS mappings
    "insights": "post insights", 
    "stats": "post insights", 
    "performance": "post insights",
    "how to view insights": "post insights",
    "how to check stats": "post insights",
    "how to see post insights": "post insights",
    "how to check post performance": "post insights",
    
    # SAVED POSTS mappings
    "bookmark": "saved posts", 
    "save post": "saved posts", 
    "saved content": "saved posts",
    "how to save post": "saved posts",
    "how to bookmark": "saved posts",
    "how to save posts": "saved posts",
    "how to bookmark posts": "saved posts",
    "how to view saved posts": "saved posts",
    
    # EDIT PROFILE mappings
    "profile": "edit profile", 
    "update profile": "edit profile", 
    "change profile": "edit profile",
    "how to edit profile": "edit profile",
    "how to update profile": "edit profile",
    "how to change profile": "edit profile",
    
    # EDIT POST mappings
    "update post": "edit post", 
    "modify post": "edit post", 
    "change post": "edit post",
    "how to edit post": "edit post",
    "how to update post": "edit post",
    "how to modify post": "edit post",
    "how to change post": "edit post",
    
    # BLOCK/UNBLOCK USER mappings
    "block": "block/unblock user", 
    "unblock": "block/unblock user", 
    "restrict user": "block/unblock user",
    "how to block": "block/unblock user",
    "how to unblock": "block/unblock user",
    "how to block user": "block/unblock user",
    "how to unblock user": "block/unblock user",
    "how to block a user": "block/unblock user",
    "how to unblock a user": "block/unblock user",
    
    # HIDE/UNHIDE USERS mappings
    "hide user": "hide/unhide users", 
    "unhide user": "hide/unhide users", 
    "hide content": "hide/unhide users",
    "how to hide": "hide/unhide users",
    "how to unhide": "hide/unhide users",
    "how to hide user": "hide/unhide users",
    "how to unhide user": "hide/unhide users",
    "how to hide a user": "hide/unhide users",
    "how to unhide a user": "hide/unhide users",
    
    # MESSAGES mappings
    "message": "messages", 
    "dm": "messages", 
    "direct message": "messages", 
    "chat": "messages",
    "how to message": "messages",
    "how to send message": "messages",
    "how to send messages": "messages",
    "how to dm": "messages",
    "how to direct message": "messages",
    "how to chat": "messages",
    
    # DISCOVERY mappings
    "discover": "discovery", 
    "explore": "discovery", 
    "find content": "discovery", 
    "search": "discovery",
    "how to discover": "discovery",
    "how to explore": "discovery",
    "how to find content": "discovery",
    "how to search": "discovery",
    
    # CREATE A PLAYLIST mappings
    "playlist": "create a playlist", 
    "series": "create a playlist", 
    "collection": "create a playlist",
    "create playlist": "create a playlist",
    "make playlist": "create a playlist",
    "how to playlist": "create a playlist",
    "how to create playlist": "create a playlist",
    "how to make playlist": "create a playlist",
    "how to create a playlist": "create a playlist",
    "how to make a playlist": "create a playlist"
}

# Direct implementation of tools as functions
def platform_guide(section: str) -> str:
    """Provides guidance about different sections of the social media platform
    Args:
        section: The platform section/feature the user needs help with
    """
    # Standardize input to match allowed sections
    std_section = section.lower()
    for mapping_key, mapping_value in CONTENT_TYPE_MAPPING.items():
        if mapping_key in std_section:
            std_section = mapping_value
            break
    
    platform_sections = {
        # Original sections
        #"profile": "To edit your profile:\n1. Click on your avatar\n2. Select 'Edit Profile'\n3. Update your information\n4. Click 'Save Changes'",
        #"messages": "To send private messages:\n1. Click the messages icon\n2. Select a contact or search for one\n3. Type your message\n4. Press enter or click send",
        #"settings": "To access settings:\n1. Click on your avatar\n2. Select 'Settings'\n3. Choose the category you want to modify",

        "shot": "SHOT is our platform's photo sharing feature. Would you like me to show you how to create a SHOT on our platform?",
        "snip": "SNIP is our platform's short video feature (similar to reels). Would you like me to show you how to create a SNIP on our platform?",
        "ssup": "SSUP is our platform's stories feature for temporary 24-hour content. Would you like me to show you how to create a SSUP on our platform?",
        "collab": "Our collaboration features let you create content with other users. Would you like me to show you how to use collaboration features on our platform?",
        "discovery": "The Discovery page helps you find trending content and creators. Would you like me to show you how to navigate the Discovery page?",
        "saved": "The Saved section lets you access content you've bookmarked. Would you like me to show you how to view your saved posts?",
        "drafts": "The Drafts section contains content you've started but haven't published yet. Would you like me to show you how to manage your drafts?",
        "notifications": "The Notifications section shows all activity related to your account. Would you like me to show you how to check your notifications?",
        "feedback": "You can provide feedback about the platform to help us improve. Would you like me to show you how to submit feedback?",
        "moments": "Moments are collections of your archived content. Would you like me to show you how to create and manage Moments?",
        "playlist": "Playlists allow you to organize multiple FLIX videos. Would you like me to show you how to create a playlist?",
        "flix": "FLIX is our platform's longer video format. Would you like me to show you how to create a FLIX?",
        "account": "Account settings let you manage your profile details. Would you like me to show you how to access account settings?",
        "insights": "Insights provide analytics about your content performance. Would you like me to show you how to view your insights?",
        "themes": "You can customize the app's appearance with different themes. Would you like me to show you how to change themes?",
        "blocking": "Blocking prevents specific users from interacting with you. Would you like me to show you how to block or unblock users?",
        "hiding": "Hiding lets you remove specific users' content from your feed. Would you like me to show you how to hide or unhide users?",
        "reporting": "Reporting helps keep the community safe by flagging inappropriate content. Would you like me to show you how to report content?",
        "editing": "Our platform offers various editing tools for your content. Would you like me to show you specific editing features?",
        "interactive": "Interactive elements make your SNIP videos more engaging. Would you like me to show you how to create interactive SNIPs?"
    }
    
    # Only return info for allowed sections
    return platform_sections.get(std_section, "I don't have information about that section. Perhaps you're interested in creating content? Try asking about 'SHOT', 'SNIP', 'SSUP', 'FLIX', or 'collab', or other platform features like 'editing', 'moments', or 'playlists'.")
    
    # Only return info for allowed sections
    return platform_sections.get(std_section, "I don't have information about that section. Perhaps you're interested in creating content? Try asking about 'SHOT', 'SNIP', 'SSUP', or 'collab'.")

def handle_common_issues(issue_type: str) -> str:
    """Handles common platform issues and provides solutions
    Args:
        issue_type: The type of issue the user is experiencing
    """
    solutions = {
        # Original issues
        "login": "If you're having trouble logging in:\n1. Check your username/password\n2. Clear Application cache\n3. Reset password if needed",
        "upload": "For upload issues:\n1. Check file size (max 20MB)\n2. Ensure supported format\n3. Check internet connection",
        "notification": "For notification problems:\n1. Check app permissions\n2. Verify notification settings\n3. Restart the app",
        "privacy": "To adjust privacy settings:\n1. Go to Settings > Privacy\n2. Choose who can see your content\n3. Save changes",
        
        # New issues
        "account": "For account issues:\n1. Verify your email is confirmed\n2. Check if your account meets community guidelines\n3. Contact support if problems persist",
        "content": "For content issues:\n1. Check your internet connection\n2. Ensure content meets guidelines\n3. Try uploading again after restarting the app",
        "technical": "For technical issues:\n1. Update to the latest app version\n2. Restart your device\n3. Clear Application cache\n4. Reinstall the app if problems persist",
        "app": "For app performance issues:\n1. Close background apps\n2. Free up device storage\n3. Update to the latest version\n4. Reinstall the app if problems persist",
        "video": "For video playback issues:\n1. Check your internet connection\n2. Clear Application cache\n3. Reduce video quality in Settings > Data Usage",
        "audio": "For audio issues:\n1. Check device volume\n2. Toggle device mute switch\n3. Check if headphones are properly connected\n4. Restart the app",
        "connection": "For connection issues:\n1. Switch between WiFi and mobile data\n2. Toggle airplane mode\n3. Restart your router\n4. Check if Bigshorts servers are down",
        "quality": "For content quality issues:\n1. Upload original high-quality files\n2. Check internet bandwidth\n3. Wait for processing to complete\n4. Adjust quality settings in the app",
        "blocking": "For blocking issues:\n1. Go to Me > hamburger menu > Blocked Users\n2. Find the user you want to unblock\n3. Tap Unblock\n4. For new blocks, go to the user's profile and select Block",
        "reporting": "For reporting issues:\n1. Find the content you want to report\n2. Tap the three dots\n3. Select Report\n4. Choose the appropriate category\n5. Add details and submit",
        "messaging": "For messaging issues:\n1. Check your internet connection\n2. Verify the user hasn't blocked you\n3. Clear chat history\n4. Restart the app",
        "password": "For password issues:\n1. Use the Forgot Password feature\n2. Check your email for reset instructions\n3. Create a strong new password\n4. Update password in all logged-in devices",
        "theme": "For theme issues:\n1. Go to Me > hamburger menu > App Theme Preference\n2. Select a different theme\n3. If theme isn't applying, restart the app\n4. Clear Application cache if problems persist"
    }
    
    # Only handle predefined issues
    if issue_type.lower() not in ALLOWED_ISSUE_TYPES:
        return "I don't have a solution for that issue. Try asking about 'login', 'upload', 'notification', 'privacy', 'account', 'content', 'technical', 'app', 'video', 'audio', 'connection', 'quality', 'blocking', 'reporting', 'messaging', 'password', or 'theme'."
    
    return solutions.get(issue_type.lower())

def generate_interactive_video_ideas() -> str:
    """Generates random interactive video ideas for viewers in Snips."""
    ideas = [
        "Create a Snip where viewers can click a button to watch a related tutorial video.",
        "Add an interactive button that redirects viewers to a behind-the-scenes video of your content creation process.",
        "Create a Snip where viewers can choose which type of content they want to watch next (e.g., comedy, tech, beauty).",
        "Embed a button that lets viewers explore different topics you've covered in your videos.",
        "Add a button that takes viewers to a Q&A session you've recorded, where they can learn more about you.",
        "Let viewers jump to a video that reveals a surprise ending or twist to your current Snip.",
        "Create a video where viewers can choose to see a blooper reel or extra footage with the click of a button.",
        "Add an interactive button that takes viewers to a related series of videos, forming a mini-series experience.",
        "Create a Snip where viewers can click a button to vote for their favorite content style and influence what you post next.",
        "Use an interactive button to send viewers to a product review or demo video linked to your current Snip's theme."
    ]
    
    selected_idea = random.choice(ideas)
    return f"Here's an interactive idea for your Snip: {selected_idea}"

def content_creation_guide(content_type: str) -> dict:
    """Provides detailed guidance about different types of content creation on Bigshorts
    Args:
        content_type: The type of content user wants to create (e.g., 'SHOT', 'SNIP', 'ssup', 'collab')
    Returns:
        dict: Contains steps, tips, and image paths for visual guidance
    """
    # Standardize input to match allowed content types
    std_content_type = content_type.lower()
    for mapping_key, mapping_value in CONTENT_TYPE_MAPPING.items():
        if mapping_key.lower() in std_content_type:
            std_content_type = mapping_value.lower()
            break
    
    # Only allow predefined content types
    if std_content_type.lower() not in [ct.lower() for ct in ALLOWED_CONTENT_TYPES]:
        return {
            "type": "content_guide",
            "content": {
                "title": "Content Type Not Found",
                "steps": [],
            }
        }

    guides_case_insensitive = {}
    
    guides = {
        "shot": {
            "title": "Creating a Bigshorts SHOT",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the Bigshorts app and tap the Creation Button",
                    "image_path": "images/Shot/Group 1449.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Choose 'SHOT' from the Creation Wheel",
                    "image_path": "images/Shot/Group 1450.png",
                    
                },
                {
                    "step": 3,
                    "description": "Capture a SHOT or upload an existing photo from your Device",
                    "image_path": "images/Shot/Group 1451.png",
                    "tips": "SHOT can include multiple Pictures"
                    
                },
                {
                    "step": 4,
                    "description": "Edit your SHOT using Bigshorts tools",
                    "image_path": "images/Shot/Group 1452.png",
                    "tips": "Try our AI-powered filters and effects"
                },
                {
                    "step": 5,
                    "description": "Add captions, hashtags, and description Or Collab with your Friends and post",
                    "image_path": "images/Shot/Group 1453.png",
                    "tips": "Use trending hashtags for better reach"
                }
            ],
        },
        "ssup": {
            "title": "Creating a Bigshorts SSUP",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the Bigshorts app and tap the Creation Button",
                    "image_path": "images/Shot/Group 1439.png",
                   
                },
                {
                    "step": 2,
                    "description": "Choose 'SSUP' from the Creation Wheel",
                    "image_path": "images/Shot/Group 1440.png",
                   
                },
                {
                    "step": 3,
                    "description": "Capture a video/image or upload an existing one from your Device",
                    "image_path": "images/Shot/Group 1441.png",
                   
                },
                {
                    "step": 4,
                    "description": "Edit your SSUP using Bigshorts tools and tap done",
                    "image_path": "images/Shot/Group 1442.png",
                    "tips": "Try our AI-powered filters and effects"
                    
                },
                {
                    "step": 5,
                    "description": "Select your desired Duration, choose who can see your SSUP and Share",
                    "image_path": "images/Shot/Group 1443.png",
                    "tips": "Choose who can see your SSUP"
                    
                }
            ],
        },
        "snip": {
            "title": "Creating a Bigshorts SNIP",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the Bigshorts app and tap the Creation Button",
                    "image_path": "images/Shot/Group 1444.png",
                    "tips": "Ensure stable internet connection"
                },
                {
                    "step": 2,
                    "description": "Choose 'SNIP' from the Creation Wheel",
                    "image_path": "images/Shot/Group 1445.png",
                },
                {
                    "step": 3,
                    "description": "Capture video or choose a video and click next",
                    "image_path": "images/Shot/Group 1446.png",
                },
                {
                    "step": 4,
                    "description": "Edit your SNIP using Bigshorts tools and tap done",
                    "image_path": "images/Shot/Group 1447.png",
                },
                {
                    "step": 5,
                    "description": "Add captions, hashtags, and description Or Collab with your Friends",
                    "image_path": "images/Shot/Group 1448.png",
                    "tips": "Use trending hashtags for better reach"
                }
            ],
        },
        "collab": {
            "title": "Creating Collaborative Content",
            "steps": [
                {
                    "step": 1,
                    "description": "While posting, Tap Collaborate with your friends to add mentions in the end.",
                    "image_path": "images/Shot/Group 1454.png",
                    "tips": "Available for creators with 1000+ followers"
                },
                {
                    "step": 2,
                    "description": "Search for a user by typing their name in the Search Mention bar, then select them from the list.",
                    "image_path": "images/Shot/Group 1455.png",
                    "tips": "Can add up to 4 collaborators"
                },
                {
                    "step": 3,
                    "description": "Once done, you can either save as a draft or post it!",
                    "image_path": "images/Shot/Group 1456.png",
                    "tips": "Clearly define each creator's role"
                },
                {
                    "step": 4,
                    "description": "On another account, To approve a collaboration, tap the Notifications button at the top.",
                    "image_path": "images/Shot/Group 1457.png",
                    "tips": "Clearly define each creator's role"
                },
                {
                    "step": 5,
                    "description": "Find the Requested to Collaborate notification, then tap Accept — and you're done! 🎉",
                    "image_path": "images/Shot/Group 1458.png",
                    "tips": "Clearly define each creator's role"
                },
                {
                    "step": 6,
                    "description": "You can see your colloborated post on scroll screen",
                    "image_path": "images/Shot/Group 1459.png",
                    "tips": "Clearly define each creator's role"
                }
            ],
        },
        "Editing a shot": {
            "title": "Editing a Bigshorts SHOT",
            "steps": [
                {
                    "step": 1,
                    "description": "Apply desired filter and adjust brightness, apart from many effects lets explore image in image (Highlighted in red)",
                    "image_path": "images/Shot/Group 1475.png",
                    "tips": "Try our AI-powered filters and effects"
                },{
                    "step": 2,
                    "description": "Choose filter of choice and select image (highlighted in red)",
                    "image_path": "images/Shot/Group 1476.png",
                    
                },{
                    "step": 3,
                    "description": "Choose the image you want, then tap on done to proceed",
                    "image_path": "images/Shot/Group 1477.png",
                    
                },{
                    "step": 4,
                    "description": "Edit the image as needed, then tap on tick mark to proceed",
                    "image_path": "images/Shot/Group 1478.png",
                    
                },{
                    "step": 5,
                    "description": "Place the image in image on screen as desiredand tap done",
                    "image_path": "images/Shot/Group 1479.png",
                    
                },{
                    "step": 6,
                    "description": "Tap done to save effect changes",
                    "image_path": "images/Shot/Group 1480.png",
                    
                },{
                    "step": 7,
                    "description": "Add captions, hashtags, and description Or Collab with your Friends and post",
                    "image_path": "images/Shot/Group 1481.png",
                }
            ]
        },
        "Invite friends": {
            "title": "Inviting friends",
            "steps": [
                {
                    "step": 1,
                    "description": "In the Me section, tap Add Friends from the top bar (highlighted in red).",
                    "image_path": "images/Shot/Group 1533.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Invite your family and friends easily!",
                    "image_path": "images/Shot/Group 1534.png",
                    
                }
            ]
        },
        "Feedback": {
            "title": "Feedback",
            "steps": [
                {
                    "step": 1,
                    "description": "Tap in the Me section and Tap the 3 lines menu at the top right corner to open settings",
                    "image_path": "images/Shot/Group 1539.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select Feedback (highlighted in red).",
                    "image_path": "images/Shot/Group 1540.png",
                    
                },
                {
                    "step": 3,
                    "description": "Fill in your necessary details",
                    "image_path": "images/Shot/Group 1541.png",
                    
                },
                {
                    "step": 4,
                    "description": "After filling you feedback/suggestions tap on Submit",
                    "image_path": "images/Shot/Group 1542.png",
                    
                }
            ]
        },
        "Multiple accounts": {
            "title": "Multiple accounts",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, you can switch or add multiple accounts by long-pressing the Me button or on top left click on your username",
                    "image_path": "images/Shot/Group 1530.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select the account by tapping the radio button or add account.",
                    "image_path": "images/Shot/Group 1531.png",
                    
                },
                {
                    "step": 3,
                    "description": "Heres your changed account Me section",
                    "image_path": "images/Shot/Group 1532.png",
                    
                }
            ]
        },
        "Account overview": {
            "title": "Account overview",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, Tap the Account Overview button at the top (highlighted in red).",
                    "image_path": "images/Shot/Group 1505.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "View and filter stats for different time periods by tapping the Filter button (highlighted in red).",
                    "image_path": "images/Shot/Group 1506.png",
                    
                },
                {
                    "step": 3,
                    "description": "Scroll to see more metrics and you can also change period for which you want a overview",
                    "image_path": "images/Shot/Group 1507.png",
                    
                }
            ]
        },
        "Store Draft": {
            "title": "Storing draft",
            "steps": [
                {
                    "step": 1,
                    "description": "Click on Save to Draft at the last stage before posting",
                    "image_path": "images/Shot/Group 1550.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "To view or post the content later, navigate to Me section (highlighted in yellow) and tap on the Draft icon (highlighted in red).",
                    "image_path": "images/Shot/Group 1551.png",
                    
                }
            ]
        },
        "Change Password": {
            "title": "Changing password",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, Tap the Hamburger menu at the top (highlighted in red) to open settings.",
                    "image_path": "images/Shot/Group 1502.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select Change Password (highlighted in red).",
                    "image_path": "images/Shot/Group 1503.png",
                    
                },
                {
                    "step": 3,
                    "description": "Enter your current password and new password, then confirm the change.",
                    "image_path": "images/Shot/Group 1504.png",
                    
                }
            ]
        },
        "Notification": {
            "title": "Viewing notifications",
            "steps": [
                {
                    "step": 1,
                    "description": "On the Home page, tap on the Notification Bell icon at the top.",
                    "image_path": "images/Shot/Group 1555.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Check out all your notifications here.",
                    "image_path": "images/Shot/Group 1556.png",
                    
                }
            ]
        },
        "Change theme": {
            "title": "Changing theme",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, Tap the Hamburger menu at the top to open settings.",
                    "image_path": "images/Shot/Group 1535.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select App Theme Preference (highlighted in red).",
                    "image_path": "images/Shot/Group 1536.png",
                    
                },
                {
                    "step": 3,
                    "description": "Choose a theme by tapping on it, based on your preference.",
                    "image_path": "images/Shot/Group 1537.png",
                    
                },
                {
                    "step": 4,
                    "description": "Your new Theme has been applied.",
                    "image_path": "images/Shot/Group 1538.png",
                    
                }
            ]
        },
        "Report": {
            "title": "Reporting a user",
            "steps": [
                {
                    "step": 1,
                    "description": "On a post, tap on the three-dots menu.",
                    "image_path": "images/Shot/Group 1543.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": " Select Report (highlighted in red).",
                    "image_path": "images/Shot/Group 1544.png",
                    
                },
                {
                    "step": 3,
                    "description": "Choose the category of the reported content and add a comment if required.",
                    "image_path": "images/Shot/Group 1545.png",
                    
                },
                {
                    "step": 4,
                    "description": "Tap Submit to finalize the report.",
                    "image_path": "images/Shot/Group 1546.png",
                    
                }
            ]
        },
        "Moment": {
            "title": "creating a moment",
            "steps": [
                {
                    "step": 1,
                    "description": "On Me page, Tap the Hamburger menu at the top to open settings.",
                    "image_path": "images/Shot/Group 1496.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select Archives (highlighted in red).",
                    "image_path": "images/Shot/Group 1497.png",
                    
                },
                {
                    "step": 3,
                    "description": "Tap the three-dots menu at the top.",
                    "image_path": "images/Shot/Group 1498.png",
                },
                {
                    "step": 4,
                    "description": "Select Create a Moment",
                    "image_path": "images/Shot/Group 1499.png",
                    
                },
                {
                    "step": 5,
                    "description": "Choose the archive(s) you want to include, add a title (highlighted in red), then tap Confirm at the top (Highlighted in yellow).",
                    "image_path": "images/Shot/Group 1500.png",
                },
                {
                    "step": 6,
                    "description": "Hooray! 🎉 Your Moment is now visible on your profile!",
                    "image_path": "images/Shot/Group 1501.png",
                    
                }
            ]
        },
        "Delete Post": {
            "title": "Deleting a post",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the post you want to delete.",
                    "image_path": "images/Shot/Group 1511.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap the three-dots menu and select Delete Shot (highlighted in red)",
                    "image_path": "images/Shot/Group 1512.png",
                    
                },
                {
                    "step": 3,
                    "description": "Tap on delete Shot",
                    "image_path": "images/Shot/Group 1513.png",
                    
                },
                {
                    "step": 4,
                    "description": "Confirm with Yes — and it's deleted!",
                    "image_path": "images/Shot/Group 1514.png",
                    
                }
            ]
        },
        "Post insights": {
            "title": "Post insights",
            "steps": [
                {
                    "step": 1,
                    "description": "On Me section, Tap on the post you want to check insights for.",
                    "image_path": "images/Shot/Group 1508.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Click on Insights (highlighted in red).",
                    "image_path": "images/Shot/Group 1509.png",
                    
                },
                {
                    "step": 3,
                    "description": "View all key metrics related to your post.",
                    "image_path": "images/Shot/Group 1510.png",
                    
                }
            ]
        },
        "Saved Posts": {
            "title": "Saving posts",
            "steps": [
                {
                    "step": 1,
                    "description": "To save a post, tap the bookmark icon below a post (highlighted in yellow), Navigate to Me (highlighted in red)",
                    "image_path": "images/Shot/Group 1526.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap on the Saved section (highlighted in red).",
                    "image_path": "images/Shot/Group 1527.png",
                    
                },
                {
                    "step": 3,
                    "description": "View all your saved photos, videos, and music.",
                    "image_path": "images/Shot/Group 1528.png",
                    
                },
                {
                    "step": 4,
                    "description": "Tap on any folder to check them out.",
                    "image_path": "images/Shot/Group 1529.png",
                    
                }
            ]
        },
        "Edit Profile": {
            "title": "Editing your Profile",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, tap Edit Profile.",
                    "image_path": "images/Shot/Group 1520.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Update your personal details and profile picture, To change or remove your profile picture, tap the pencil icon on the image.",
                    "image_path": "images/Shot/Group 1521.png",
                    
                },
                {
                    "step": 3,
                    "description": "Choose to take a photo or select one from your gallery or remove photo",
                    "image_path": "images/Shot/Group 1522.png",
                    
                },
                {
                    "step": 4,
                    "description": "Choose desired photo",
                    "image_path": "images/Shot/Group 1523.png",
                    
                },
                {
                    "step":5,
                    "description": "Rotate, Crop and adjust the image, then tap the tick icon at the top (highlighted in red) to save changes.",
                    "image_path": "images/Shot/Group 1524.png",
                    
                },
                {
                    "step":6,
                    "description": "Finally, save your profile by clicking on top right save button.",
                    "image_path": "images/Shot/Group 1525.png",
                    
                }
            ]
        },
        "Edit Post": {
            "title": "Editing a post",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the post you want to edit.",
                    "image_path": "images/Shot/Group 1515.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap the three-dot menu.",
                    "image_path": "images/Shot/Group 1516.png",
                    
                },
                {
                    "step": 3,
                    "description": "Select “Edit Shot” (highlighted in red).",
                    "image_path": "images/Shot/Group 1517.png",
                    
                },
                {
                    "step": 4,
                    "description": "Make the necessary changes like change description, add collab, set who can watch the post, change location",
                    "image_path": "images/Shot/Group 1518.png",
                    
                },
                {
                    "step": 4,
                    "description": "Then tap Update Post — done!",
                    "image_path": "images/Shot/Group 1519.png",
                    
                }
            ]
        },
        "Block/unblock User": {
            "title": "Blocking and unblocking users",
            "steps": [
                {
                    "step": 1,
                    "description": "On the selected user’s post, tap the three-dot menu.",
                    "image_path": "images/Shot/Group 1482.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select Block User — and you're done! 🚫",
                    "image_path": "images/Shot/Group 1483.png",
                    
                },
                {
                    "step": 3,
                    "description": "Go to Me from the bottom navigation bar and click on 3 lines on top right.",
                    "image_path": "images/Shot/Group 1484.png",
                    
                },
                {
                    "step": 4,
                    "description": "Select Blocked Users (highlighted in red).",
                    "image_path": "images/Shot/Group 1485.png",
                    
                },
                {
                    "step": 5,
                    "description": "Here, you can view all blocked users or unblock them if needed.",
                    "image_path": "images/Shot/Group 1486.png",
                    
                }
            ]
        },
        "Hide/Unhide Users": {
            "title": "Hiding and unhiding users",
            "steps": [
                {
                    "step": 1,
                    "description": "On the selected user's post, tap the three-dot menu.",
                    "image_path": "images/Shot/Group 1491.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap Hide Content (highlighted in red), and the user's content is Hidden!.",
                    "image_path": "images/Shot/Group 1492.png",
                    
                },
                {
                    "step": 3,
                    "description": "Go to Me section from the bottom navigation bar, and Tap the Hamburger menu at the top.",
                    "image_path": "images/Shot/Group 1493.png",
                    
                },
                {
                    "step": 4,
                    "description": "Select Hidden Users.",
                    "image_path": "images/Shot/Group 1494.png",
                    
                },
                {
                    "step": 5,
                    "description": "Here, you can view the list of hidden users and unhide them if needed.",
                    "image_path": "images/Shot/Group 1495.png",
                    
                }
            ]
        },
        "Messages": {
            "title": "Messaging a user",
            "steps": [
                {
                    "step": 1,
                    "description": "Click on the message icon on the top right corner to send a message ",
                    "image_path": "images/Shot/Group 1552.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Choose the person you want to message ",
                    "image_path": "images/Shot/Group 1553.png",
                    
                },
                {
                    "step": 3,
                    "description": "To attach(image or videos) in your dm click on the attach icon or you can also record audio and send",
                    "image_path": "images/Shot/Group 1554.png",
                }
            ]
        },
        "Discovery": {
            "title": "Navigating discovery page",
            "steps": [
                {
                    "step": 1,
                    "description": "Tap scroll icon on the bottom navigation bar (Highlighted in red)",
                    "image_path": "images/Shot/Group 1487.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Then click search icon on top bar (Highlighted in red)",
                    "image_path": "images/Shot/Group 1488.png",
                    
                },
                {
                    "step": 3,
                    "description": "Here you can view famous hashtags accordingly on discovery tab on top, to search tap on search icon",
                    "image_path": "images/Shot/Group 1489.png",
                    
                },
                {
                    "step": 4,
                    "description": "You can search users, hashtag in our search barr",
                    "image_path": "images/Shot/Group 1490.png",
                    
                }
            ]
        },
        "Editing a Ssup": {
            "title": "Editing a BigShorts Ssup",
            "steps": [
                {
                    "step": 1,
                    "description": "Click on the edit button, after choosing a content to upload",
                    "image_path": "images/Shot/Group 1468.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "You can adjust(Brightness, contrast, saturation, sharpness), retouch, makeup and add effects or text and then click on the Save button",
                    "image_path": "images/Shot/Group 1469.png",
                    
                },
                {
                    "step": 3,
                    "description": "You can also add various effects (like sticker, filter, location, links, image in images, etc), lets explore music",
                    "image_path": "images/Shot/Group 1470.png",
                    
                },{
                    "step": 4,
                    "description": "Select the music you want",
                    "image_path": "images/Shot/Group 1471.png",
                },
                {
                    "step": 5,
                    "description": "Choose the portion of the music and click Apply sound",
                    "image_path": "images/Shot/Group 1472.png",
                },
                {
                    "step": 6,
                    "description": "After you have applied your desired effects click on done",
                    "image_path": "images/Shot/Group 1473.png",
                },
                {
                    "step": 7,
                    "description": "Select your desired Duration, choose who can see your SSUP and Share",
                    "image_path": "images/Shot/Group 1474.png",
                }
            ]
        },
        "Interactive snip": {
            "title": "Making an Interactive Snip",
            "steps": [
                {
                    "step": 1,
                    "description": "While editing a snip, add button(highlighted in red) to add interactive elements",
                    "image_path": "images/Shot/Group 1592.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Edit the button as needed",
                    "image_path": "images/Shot/Group 1593.png",
                    
                },
                {
                    "step": 3,
                    "description": "Click on the interactive tap button (highlighted in red) to add more interactive elements",
                    "image_path": "images/Shot/Group 1594.png",
                    
                },{
                    "step": 4,
                    "description": "Select a type of interactive element",
                    "image_path": "images/Shot/Group 1595.png",
                },
                {
                    "step": 5,
                    "description": "Capture a snip or select from gallery",
                    "image_path": "images/Shot/Group 1596.png",
                    
                },
                {
                    "step": 6,
                    "description": "Click on timeline, to edit interactive duration",
                    "image_path": "images/Shot/Group 1597.png",
                    
                },{
                    "step": 7,
                    "description": "You can view and adjust the interactive elements timeline here",
                    "image_path": "images/Shot/Group 1598.png",
                },
                {
                    "step": 8,
                    "description": "Click on interactive tree hierarchy (highlighted in red)",
                    "image_path": "images/Shot/Group 1599.png",
                    
                },
                {
                    "step": 9,
                    "description": "Here you can view hierarchy tree of interactive elements",
                    "image_path": "images/Shot/Group 1600.png",
                    
                },
                {
                    "step": 10,
                    "description": "Click on post to publish your interactive video.",
                    "image_path": "images/Shot/Group 1601.png",
                    
                }
            ]
        },
        "Flix": {
            "title": "Creating a flix",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the Bigshorts app and tap the Creation Button",
                    "image_path": "images/Shot/Group 1557.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Choose 'FLIX' from the Creation Wheel",
                    "image_path": "images/Shot/Group 1558.png",
                    
                },
                {
                    "step": 3,
                    "description": "Capture a Flix or upload an existing one from your Device and click next",
                    "image_path": "images/Shot/Group 1559.png",
                    
                },{
                    "step": 4,
                    "description": "Pick a cover image for your Flix, add description, title, allow comment or who can watch the flix",
                    "image_path": "images/Shot/Group 1560.png",
                    
                },{
                    "step": 5,
                    "description": "After filling the fields tap on post",
                    "image_path": "images/Shot/Group 1561.png",
                    
                },{
                    "step": 6,
                    "description": "After it, tap on Post and you're done!",
                    "image_path": "images/Shot/Group 1562.png",
                    
                }
            ]
        },
        "Mini Drama series": {
            "title": "Creating a Mini Drama series",
            "steps": [
                {
                    "step": 1,
                    "description": "On Me section, Tap on the Create Mini Drama Series",
                    "image_path": "images/Shot/Group 1576.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Select a cover image for the Mini Drama series.",
                    "image_path": "images/Shot/Group 1577.png",
                    
                },
                {
                    "step": 3,
                    "description": "Choose a image from gallery",
                    "image_path": "images/Shot/Group 1578.png",
                    
                },
                {
                    "step": 4,
                    "description": "Edit the cover image that you selected and then save",
                    "image_path": "images/Shot/Group 1579.png",
                    
                },
                {
                    "step": 5,
                    "description": "Add season title, description and eventually schedule time.",
                    "image_path": "images/Shot/Group 1580.png",
                    
                },
                {
                    "step": 6,
                    "description": "And then click on Create Mini drama series (highlighted in red)",
                    "image_path": "images/Shot/Group 1581.png",
                    
                },
                {
                    "step": 7,
                    "description": "Select desired number of flix that you want to add to Mini Drama series",
                    "image_path": "images/Shot/Group 1582.png",
                    
                },
                {
                    "step": 8,
                    "description": "And then tap on Add Episodes",
                    "image_path": "images/Shot/Group 1583.png",
                    
                },
                {
                    "step": 9,
                    "description": "Viola! your mini drama series is created!",
                    "image_path": "images/Shot/Group 1584.png",
                    
                }
            ]
        },
        "Editing a flix": {
            "title": "Editing a BigShorts flix",
            "steps": [
                {
                    "step": 1,
                    "description": "Open the Bigshorts app and tap the Creation Button",
                    "image_path": "images/Shot/Group 1563.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Choose 'FLIX' from the Creation Wheel",
                    "image_path": "images/Shot/Group 1564.png",
                    
                },
                {
                    "step": 3,
                    "description": "Capture a Flix or upload an existing one from your Device and click next",
                    "image_path": "images/Shot/Group 1565.png",
                    
                },
                {
                    "step": 4,
                    "description": "When you chose a video to upload, next you can edit it by Rotate, split, trimming or deleted a splitted clip then tap on tick mark",
                    "image_path": "images/Shot/Group 1566.png",
                   
                },
                {
                    "step": 5,
                    "description": "Tap on record to add your voice or any other sound",
                    "image_path": "images/Shot/Group 1567.png",
                    
                },
                {
                    "step": 6,
                    "description": "Tap on Record button",
                    "image_path": "images/Shot/Group 1568.png",
                    
                },{
                    "step": 7,
                    "description": "After recording click on tick mark",
                    "image_path": "images/Shot/Group 1569.png",
                    
                },
                {
                    "step": 8,
                    "description": "You can also add sound effect on your recording",
                    "image_path": "images/Shot/Group 1570.png",
                    
                },
                {
                    "step": 9,
                    "description": "Select a effect and apply",
                    "image_path": "images/Shot/Group 1571.png",
                    
                },{
                    "step": 10,
                    "description": "Then tap on tick",
                    "image_path": "images/Shot/Group 1572.png",
                    
                },
                {
                    "step": 11,
                    "description": "Click on Time effect in effects board",
                    "image_path": "images/Shot/Group 1573.png",
                    
                },
                {
                    "step": 12,
                    "description": "Tap and hold on desired time effect to apply, then tap on tick mark",
                    "image_path": "images/Shot/Group 1574.png",
                    
                },
                {
                    "step": 13,
                    "description": "Click Next to save your effects, to proceed to posting",
                    "image_path": "images/Shot/Group 1575.png",
                    
                }
            ]
        },
        "Editing a Snip": {
            "title": "Editing a BigShorts Snip",
            "steps": [
                {
                    "step": 1,
                    "description": "When uploaded a snip, you can cick on edit for video editing",
                    "image_path": "images/Shot/Group 1460.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "You can Rotate, split, trim, or delete clips and tap on tick mark to proceed to effects board",
                    "image_path": "images/Shot/Group 1461.png",
                    
                },
                {
                    "step": 3,
                    "description": "Click on Blur amongst many effects which you can choose to edit",
                    "image_path": "images/Shot/Group 1462.png",
                    
                },
                {
                    "step": 4,
                    "description": "Select a desired shape of blur and apply blur effect",
                    "image_path": "images/Shot/Group 1463.png",
                    
                },{
                    "step": 5,
                    "description": "Click on the next button to save effects.",
                    "image_path": "images/Shot/Group 1464.png",
                    
                },{
                    "step": 6,
                    "description": "Here you pick cover image and tap done",
                    "image_path": "images/Shot/Group 1465.png",
                   
                },
                {
                    "step": 7,
                    "description": "Click on Done to save changes",
                    "image_path": "images/Shot/Group 1466.png",
                    
                },
                {
                    "step": 8,
                    "description": "Add captions, hashtags, and description Or Collab with your Friends and post",
                    "image_path": "images/Shot/Group 1467.png",
                    
                }
            ]
        },
        "Ssup Repost": {
            "title": "Ssup Repost",
            "steps": [
                {
                    "step": 1,
                    "description": "Open your friends chats, who have mentioned you, and tap add to ssup (Shown beside mentioned story)",
                    "image_path": "images/Shot/Group 1547.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "You can apply effects and then tap post",
                    "image_path": "images/Shot/Group 1548.png",
                    
                },
                {
                    "step": 3,
                    "description": "Select your desired Duration, choose who can see your SSUP and Share",
                    "image_path": "images/Shot/Group 1549.png",
                    
                }
            ]
        },
        "Edit Mini Drama series": {
            "title": "Editing a Mini Drama series",
            "steps": [
                {
                    "step": 1,
                    "description": "In Me section, Tap on the Mini Drama Series icon (highlighted in yellow), and tap on the mini drama series you want to edit (highlighted in red)",
                    "image_path": "images/Shot/Group 1585.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap on three dots on top right corner",
                    "image_path": "images/Shot/Group 1586.png",
                    
                },
                {
                    "step": 3,
                    "description": "Tap on edit on Mini Drama Series",
                    "image_path": "images/Shot/Group 1587.png",
                    
                },
                {
                    "step": 4,
                    "description": "Edit the cover image, season title or description and click on Edit Mini Drama Series",
                    "image_path": "images/Shot/Group 1588.png",
                    
                },
                {
                    "step": 5,
                    "description": "Select new flix to be added in series or remove one.",
                    "image_path": "images/Shot/Group 1589.png",
                    
                },
                {
                    "step": 6,
                    "description": "Then tap on + Add episodes",
                    "image_path": "images/Shot/Group 1590.png",
                    
                },
                {
                    "step": 6,
                    "description": "Your series  is edited!",
                    "image_path": "images/Shot/Group 1591.png",
                    
                }
            ]
        },
        "Requested Message": {
            "title": "Requested Message",
            "steps": [
                {
                    "step": 1,
                    "description": "Open messages screen and tap on Requested (highlighted in red)",
                    "image_path": "images/Shot/Group 1602.png",
                    "tips": "Make sure you're on the latest app version for all features"
                },
                {
                    "step": 2,
                    "description": "Tap on the requesed user chat",
                    "image_path": "images/Shot/Group 1603.png",
                    
                },
                {
                    "step": 3,
                    "description": "Choose to approve or deny the user's chat",
                    "image_path": "images/Shot/Group 1604.png",
                    
                },
                {
                    "step": 4,
                    "description": "If accepted, you can chat with them now onwards!",
                    "image_path": "images/Shot/Group 1605.png",
                    
                }
            ]
        }
    }
    

    for key, value in guides.items():
        guides_case_insensitive[key.lower()] = value
    
    # CRITICAL FIX: Look up key by its lowercase version
    # This handles cases where "editing a flix" in the request
    # needs to match "Editing a flix" in the guides dictionary
    lookup_key = std_content_type.lower()
    
    # Debug output if needed
    print(f"Looking for: {lookup_key}")
    print(f"Available keys: {list(guides_case_insensitive.keys())}")
    
    # Look up the guide using the lowercase key
    guide = guides_case_insensitive.get(lookup_key)


    if guide is None:
        return {
            "type": "content_guide",
            "content": {
                "title": f"Guide for {std_content_type} not found",
                "steps": [],
            }
        }
    
    # Return in the format expected by the frontend
    return {
        "type": "content_guide",
        "content": guide
    }



def display_creation_steps(content_type: str) -> dict:
    """Generate a guide for content creation steps in a format suitable for frontend rendering
    Args:
        content_type: The type of content being created
    Returns:
        dict: Contains type and content for the ChatbotResponse interface
    """
    # Standardize input to match allowed content types
    std_content_type = content_type.lower()
    for mapping_key, mapping_value in CONTENT_TYPE_MAPPING.items():
        if mapping_key in std_content_type:
            std_content_type = mapping_value
            break
    
    # Only allow predefined content types
    if std_content_type not in ALLOWED_CONTENT_TYPES:
        return {
            "type": "error",
            "content": "Content type not found. Please try 'SHOT', 'SNIP', 'SSUP', or 'Collab'."
        }
    
    guide = content_creation_guide(std_content_type)["content"]
    
    if not guide["steps"]:
        return {
            "type": "error",
            "content": "Content type not found. Please try 'SHOT', 'SNIP', 'SSUP', or 'Collab'."
        }
    
    return {
        "type": "content_guide",
        "content": guide
    }

def detect_content_type(query: str) -> str:
    """Detects if the user query is related to platform content types
    Args:
        query: The user's message
    Returns:
        str: Detected content type or "none"
    """
    query_lower = query.lower()
    
    # Sort content types by length (descending) to prioritize more specific matches
    sorted_content_types = sorted(ALLOWED_CONTENT_TYPES, key=len, reverse=True)
    
    # Check for direct mentions of content types first (prioritizing longer, more specific types)
    for content_type in sorted_content_types:
        if content_type.lower() in query_lower:
            return content_type
    
    # Create a sorted list of keywords by length (descending) to prioritize more specific matches
    sorted_keywords = sorted(CONTENT_TYPE_MAPPING.keys(), key=len, reverse=True)
    
    # Check for related terms (prioritizing longer, more specific terms)
    for keyword in sorted_keywords:
        if keyword.lower() in query_lower:
            return CONTENT_TYPE_MAPPING[keyword]
    
    return "none"

def fallback_response() -> str:
    """Provides a standard fallback response when user query doesn't match defined areas"""
    return "I can help you with creating content like SHOT, SNIP, SSUP, or Collab, as well as handling common platform issues. What would you like help with today?"

def get_off_topic_response() -> str:
    """Returns a standardized response for off-topic queries"""
    responses = [
        "I'm your Bigshorts assistant! I can help you create amazing content (SHOT, SNIP, SSUP, Collab), troubleshoot any platform issues, or discover trending content. What would you like to explore today?",
        
        "Let's focus on making your Bigshorts experience amazing! I can guide you through creating content, solve platform issues, or show you what's trending. How can I enhance your Bigshorts journey today?",
        
        "Welcome to Bigshorts support! I'm here to help you create standout content, fix any platform issues, or discover what's trending. What aspect of Bigshorts would you like assistance with?",
        
        "As your Bigshorts assistant, I can help you create stunning SHOT photos, viral SNIP videos, engaging SSUP stories, or collaborative content. I can also troubleshoot any platform issues. What interests you most?"
    ]
    
    # Return a random response from the list for variety
    return random.choice(responses)

def get_trending_content() -> dict:
    """Returns trending content on the platform (Snips, Creators, Shots)
    Returns:
        dict: Contains trending Snips, Creators, and Shots
    """
    # Mock data - in a real implementation, this would fetch from a database
    trending_content = {
        "trending_snips": [
            {"id": "snip1", "title": "Morning Routine", "creator": "FitnessPro", "views": 1200000},
            {"id": "snip2", "title": "Easy Recipe Hack", "creator": "ChefMaster", "views": 980000},
            {"id": "snip3", "title": "Makeup Tips", "creator": "BeautyGuru", "views": 875000}
        ],
        "trending_creators": [
            {"id": "creator1", "name": "TechWhiz", "followers": 2500000, "content_type": "tech reviews"},
            {"id": "creator2", "name": "FashionForward", "followers": 2100000, "content_type": "fashion"},
            {"id": "creator3", "name": "TravelBug", "followers": 1800000, "content_type": "travel vlogs"}
        ],
        "trending_shots": [
            {"id": "shot1", "title": "Sunset Beach", "creator": "NaturePhotographer", "likes": 350000},
            {"id": "shot2", "title": "City Lights", "creator": "UrbanShots", "likes": 310000},
            {"id": "shot3", "title": "Mountain Peaks", "creator": "AdventureSeeker", "likes": 290000}
        ]
    }
    
    return trending_content

def suggest_trending_content(content_type: str = "all") -> dict:
    """Suggests trending content to the user
    Args:
        content_type: Type of content to suggest ("snips", "creators", "shots", or "all")
    Returns:
        dict: Contains buttons for frontend rendering
    """
    trending_data = get_trending_content()
    
    if content_type.lower() not in ["snips", "creators", "shots", "all"]:
        content_type = "all"
    
    buttons = []
    
    if content_type.lower() == "snips" or content_type.lower() == "all":
        buttons.append({
            "text": "Check Trending Snips",
            "action": "redirect",
            "destination": "/trending/snips"
        })
    
    if content_type.lower() == "creators" or content_type.lower() == "all":
        buttons.append({
            "text": "Discover Popular Creators",
            "action": "redirect",
            "destination": "/trending/creators"
        })
    
    if content_type.lower() == "shots" or content_type.lower() == "all":
        buttons.append({
            "text": "See Trending Shots",
            "action": "redirect",
            "destination": "/trending/shots"
        })
    
    return {
        "type": "suggestion_buttons",
        "content": {
            "message": "Check out what's trending on Bigshorts! 📈",
            "buttons": buttons
        }
    }

# Integrating all tools into a cohesive chatbot with local LLM
class BigshortsChatbot:
    def __init__(self, model_path):
        """Initialize the chatbot with a local LLM model"""
        print(f"Loading model from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Larger context size for better conversations
            n_gpu_layers=0,
            n_threads=8, 
            n_batch=4096, 
            use_mlock=True,  # Lock model in RAM
            use_mmap=False,
            prefetch=True,
            top_k=40, top_p=0.9,
            temperature=0.5,# Use GPU acceleration if available
            verbose=False
        )
        print("Model loaded successfully!")
        
        # Load custom prompt templates
        try:
            with open("prompts.yaml", 'r') as stream:
                self.prompt_templates = yaml.safe_load(stream)
        except (FileNotFoundError, yaml.YAMLError):
            print("Warning: prompts.yaml not found or invalid, using default prompts")
            self.prompt_templates = {
                "final_answer": {
                    "pre_messages": "You are a helpful social media assistant for the Bigshorts platform. Focus on helping users with platform features.",
                    "post_messages": "Remember to never show your reasoning or thought process to the user."
                }
            }
        
        self.sessions = {}
        
        # Define off-topic keywords
        self.off_topic_keywords = [
            "politics", "news", "weather", "sports", "dating", "games", "gaming", 
            "stock", "investment", "medical", "health", "drugs", "violence",
            "who is", "what is", "how many", "where is", "when did", "why does",
            "history", "science", "math", "religion", "war", "climate", "economy",
            "celebrity", "actor", "singer", "movie", "book", "covid", "virus",
            "recipe", "food", "diet", "exercise", "workout", "travel", "vacation",
            "website", "president", "election", "government", "tax", "taxes",
            "credit", "loan", "insurance", "legal", "law", "crime", "police"
        ]
        
        # Define predefined responses
        self.unsupported_query_response = {
            "type": "error",
            "content": "I can only help with Bigshorts platform features like creating SHOT, SNIP, SSUP or Collab content, and handling common issues. How can I assist you with the platform?"
        }

        
        self.content_explanations = {
            # Original content types
            "shot": "SHOT is our platform's photo content format. It lets you share pictures and photo collections with your followers.",
            "snip": "SNIP is our platform's short-form video content, similar to reels on other platforms. It's perfect for creating engaging short videos.",
            "ssup": "SSUP is our platform's stories feature - temporary content that disappears after 24 hours, perfect for quick updates and daily moments.",
            "collab": "Collaborative content allows you to create content together with other creators on our platform.",
    
            # New content types
            "editing a shot": "Our platform offers powerful tools to edit your SHOT photos, including filters, effects, adjustments, and more.",
            "invite friends": "You can easily invite friends to join you on Bigshorts and grow your network.",
            "feedback": "We value your input! You can submit feedback about the platform to help us improve.",
            "multiple accounts": "Bigshorts allows you to manage multiple accounts and easily switch between them.",
            "account overview": "Account overview provides analytics and statistics about your Bigshorts performance.",
            "store draft": "The draft feature lets you save content you're working on to finish and publish later.",
            "change password": "You can easily update your password to keep your account secure.",
            "notification": "Notifications keep you updated about activities related to your account and content.",
            "change theme": "Personalize your Bigshorts experience by choosing from different app themes.",
            "report": "The reporting feature helps maintain community standards by flagging inappropriate content.",
            "moment": "Moments let you curate and showcase collections of your archived content on your profile.",
            "delete post": "You can remove any of your content from the platform if you no longer want it visible.",
            "post insights": "Insights provide detailed analytics about how your individual posts are performing.",
            "saved posts": "You can bookmark content you like to easily find and revisit it later.",
            "edit profile": "Profile editing lets you customize your bio, avatar, and other public information.",
            "edit post": "You can modify your existing posts to update captions, tags, or other details.",
            "block/unblock user": "Blocking prevents specific users from interacting with you or seeing your content.",
            "hide/unhide users": "Hiding users removes their content from your feed without blocking them completely.",
            "messages": "Our direct messaging system lets you chat privately with other Bigshorts users.",
            "discovery": "The discovery page helps you find new content, creators, and trending topics.",
            "editing a ssup": "You can enhance your SSUP stories with various editing tools, effects, and interactive elements.",
            "interactive snip": "Interactive SNIPs allow viewers to engage with your videos through buttons and other clickable elements.",
            "flix": "FLIX is our platform's longer-form video format, perfect for more in-depth content.",
            "create a playlist": "Playlists let you organize multiple FLIX videos into collections for your audience.",
            "editing a flix": "Our FLIX editing tools help you create professional-quality longer videos.",
            "editing a snip": "SNIP editing features let you create polished, engaging short-form videos.",
            "moment": "Moments are collections of your archived SSUPs (stories) that you can showcase permanently on your profile - similar to Story Highlights on other platforms. They let you group and save your temporary SSUP content into themed collections that won't disappear after 24 hours."
        }
    
    def format_history(self, session_id):
        """Format conversation history for the LLM prompt for a specific session"""
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        formatted = ""
        # Use last 3 exchanges to save context
        conversation_history = self.sessions[session_id]
        for entry in conversation_history[-3:]:
            if entry["role"] == "user":
                formatted += f"User: {entry['content']}\n"
            else:
                # Convert dict responses to string representation if needed
                content = entry["content"]
                if isinstance(content, dict):
                    if content.get("type") == "content_guide":
                        content = f"I provided a guide for {content.get('content', {}).get('title', 'content creation')}."
                    elif content.get("type") == "message":
                        content = content.get("content", "")
                    else:
                        content = str(content)
                formatted += f"Assistant: {content}\n"
        return formatted

    
    def _is_off_topic(self, query: str) -> bool:
        """Check if query is off-topic"""
        # First check for explicit off-topic keywords
        if any(keyword in query.lower() for keyword in self.off_topic_keywords):
            return True
        
        # Then check for on-topic indicators
        on_topic_indicators = ALLOWED_CONTENT_TYPES + ALLOWED_ISSUE_TYPES + ALLOWED_PLATFORM_SECTIONS + [
            "bigshorts", "platform", "app", "create", "upload", "share", "post"
        ]
        
        if any(indicator in query.lower() for indicator in on_topic_indicators):
            return False
        
        # If query is very short (1-2 words) and doesn't contain platform terms, 
        # it might be an ambiguous command - treat as on-topic and let fallback handle it
        words = query.split()
        if len(words) <= 2:
            return False
        
        # Default to considering longer queries without platform terms as off-topic
        return True
    
    def _extract_issue(self, query: str) -> str:
        """Extract the issue type from user query - with expanded issue types"""
        for issue in ALLOWED_ISSUE_TYPES:
            if issue in query.lower():
                return issue
            
        # Check for additional issue keywords that map to standard issues
        issue_keywords = {
            "login": ["sign in", "can't log in", "login failed", "authentication", "account access"],
            "upload": ["can't upload", "upload failed", "posting problem", "sharing issue", "file problem"],
            "notification": ["alerts", "not getting notifications", "notification settings", "push notifications"],
            "privacy": ["who can see", "visibility", "hidden", "public", "private", "settings"],
            "account": ["profile", "username", "email", "verification", "account locked"],
            "payment": ["billing", "purchase", "subscription", "transaction", "payment failed"],
            "technical": ["app crash", "freezing", "not loading", "error message", "bug"],
            "video": ["playback", "buffering", "video quality", "can't play videos"],
            "audio": ["sound", "volume", "no audio", "can't hear", "music"],
            "connection": ["offline", "internet", "wifi", "data", "connectivity"],
            "password": ["forgot password", "reset password", "change password", "password reset"],
            "theme": ["dark mode", "light mode", "appearance", "display", "color scheme"]
        }
    
        for issue_type, keywords in issue_keywords.items():
            if any(keyword in query.lower() for keyword in keywords):
                return issue_type
            
        return "unknown"
    
    def _clean_agent_response(self, response: str) -> str:
        """Clean responses to remove internal processing markers"""
        # If response is already a dictionary or other object, convert to string
        response_str = str(response)
        
        # Remove typical agent process markers
        patterns_to_remove = [
            r'Thoughts?:.*?(?=Code:|Observation:|$)',
            r'Code:.*?(?=<end_code>|Observation:|$)',
            r'<end_code>',
            r'Observation:.*?(?=Thoughts?:|$)',
            r'Calling tools?:.*?(?=\n|$)',
            r'Tool call results?:.*?(?=\n|$)'
        ]
        
        cleaned = response_str
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # If we've stripped everything, use a fallback response
        if not cleaned.strip():
            return "I can help you with Bigshorts platform features. What would you like to know about SHOT, SNIP, SSUP, or Collab content?"
        
        # Remove any extra whitespace and normalize
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = re.sub(r'Assistant:', '', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

    def _is_user_search_query(self, query: str) -> bool:
        """Check if query is looking for a specific user"""
        query_lower = query.lower()
    
        # Patterns that indicate looking for a user
        user_search_patterns = [
            "@", "find user", "search user", "find profile", "search profile",
            "look for user", "find someone", "search for", "looking for"
        ]
    
        return any(pattern in query_lower for pattern in user_search_patterns)

    def generate_llm_response(self, query: str, session_id: str) -> str:
        """Generate a response using the local LLM for a specific session"""
        # Get system prompt
        system_prompt = self.prompt_templates.get("system_prompt", "")
    
        # Fallback if needed
        if not system_prompt:
            system_prompt = self.prompt_templates.get("final_answer", {}).get("pre_messages", "")
    
        # Get session-specific history
        history = self.format_history(session_id)
    
        # Format prompt with conversation history for context (Mistral format)
        prompt = f"<s>[INST] {system_prompt}\n\nConversation history:\n{history}\n\nUser's question: {query}\n\nProvide a helpful response about the Bigshorts platform: [/INST]"
    
        try:
            # Generate response with the model
            result = self.llm(
                prompt,
                max_tokens=128,
                temperature=0.5,
                stop=["</s>", "[INST]", "User:", "Human:"]
            )
        
            # Extract and clean response
            response = result["choices"][0]["text"].strip()
            response = self._clean_agent_response(response)
        
            return response
        
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return f"I encountered a technical issue. Can I help you with creating content on Bigshorts instead?"
            
    
    def process_query(self, user_input: str) -> Union[str, dict]:
        """Process user queries and return response with optional visual guide"""
        # Add user message to conversation history

        session_id = "default"
        
        if session_id is None:
            session_id = "default"
            
        # Create session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        # Add user message to conversation history
        self.sessions[session_id].append({"role": "user", "content": user_input})
    
        greetings = [
        "hello", "hi", "hey", "greetings", "howdy", "wassup", "whats up", "yo", 
        "sup", "hiya", "heya", "hola", "bonjour", "ciao", "g'day", "good morning", 
        "good afternoon", "good evening", "good day", "evening", "morning", 
        "afternoon", "hello there", "hi there", "hey there", "what's happening", 
        "what's good", "how are you", "how's it going", "how are things", 
        "how's everything", "what's new", "what's up", "sup", "yo yo", "aloha"
        ]

        if user_input.lower().strip() in greetings:
            greeting_responses = [
                "Hello! 😀 Welcome to Bigshorts! Ready to create some awesome content today?",
                "Hey there! 😃 The Bigshorts community has been buzzing with creativity. What would you like to create today?",
                "Hi! 😊 Looking to make a SHOT, SNIP, SSUP, or Collab on Bigshorts today?",
                "Greetings! 👋 Your Bigshorts assistant is ready to help you shine on the platform!",
                "Wassup! 😎 Ready to level up your Bigshorts content? I can help with SHOT, SNIP, SSUP, or Collab!",
                "Hey! 🚀 Trending content on Bigshorts is getting millions of views today. Want to create something awesome?",
                "Hello there! 🤗 What type of Bigshorts content are you looking to create today?",
                "Hi! ✨ Your Bigshorts creative journey starts here - what can I help you with?",
                "Hey! 🔥 The best Bigshorts creators start with great ideas. Need help creating your next viral content?",
                "What's up! 🎬 Bigshorts is waiting for your amazing content. Need help getting started?",
                "Yo! 🎤 Ready to make some fire content on Bigshorts? I'm here to help!",
                "Hiya! 🎉 Bigshorts creators are killing it today! Want to join them?",
                "G'day! 🌞 Let's make your Bigshorts profile stand out with some amazing content!",
                "Good morning! ☀️ Start your day with some fresh Bigshorts content creation!",
                "Good afternoon! 🌤 Perfect time to create some Bigshorts content that will trend tonight!",
                "Good evening! 🌙 Night time is prime time for Bigshorts engagement. Need help creating content?",
                "Howdy! 🤠 Your Bigshorts creative partner is here to assist with any content needs!",
                "Bonjour! 🇫🇷 Bigshorts is going global, and I'm here to help you create content that connects!",
                "Aloha! 🌺 Bring some sunshine to Bigshorts with your next SHOT, SNIP, SSUP, or Collab!",
                "Heya! 🎨 The Bigshorts algorithm loves fresh content. What would you like to create today?",
                "Sup! 🏆 Bigshorts is all about authentic content. Need help making yours stand out?",
                "How's it going? 💡 Ready to explore some creative ideas for your next Bigshorts post?",
                "What's happening! 🚀 Bigshorts is buzzing today. Let's get your content in the mix!",
                "How are you? 💬 However you're feeling, expressing it through Bigshorts content can connect with others!",
                "Ciao! 🇮🇹 Style and substance make the best Bigshorts content. Need help with either?",
                "What's good! 🏅 The best Bigshorts creators post consistently. Ready to plan your next content piece?",
                "Hi there! 🎭 Discover what's trending on Bigshorts or create something completely new!",
                "Hey hey! 🎬 Your Bigshorts assistant is ready to help with SHOT photos, SNIP videos, SSUP stories, or Collabs!",
                "Yo yo! 🚀 Bigshorts creators are changing the game! Want to join the revolution?",
                "How are things? 🛠 Whether you need help with Bigshorts creation or troubleshooting, I've got you covered!"
            ]

            faqs = [
                {
                    "question": "How do I create a SHOT?",
                    "content_type": "shot",
                    "query": "How to create a shot"
                },
                {
                    "question": "How do I create a SNIP?",
                    "content_type": "snip",
                    "query": "How to create a snip"
                },
                {
                    "question": "How do I create a SSUP?",
                    "content_type": "ssup",
                    "query": "How to create a ssup"
                },
                {
                    "question": "How do I make a Collab post?",
                    "content_type": "collab",
                    "query": "How to collaborate"
                },
                {
                    "question": "How do I edit my profile?",
                    "content_type": "edit profile",
                    "query": "How to edit profile"
                },
                {
                    "question": "How do I see my notifications?",
                    "content_type": "notification",
                    "query": "How to check notifications"
                },
                {
                    "question": "How do I change the app theme?",
                    "content_type": "change theme",
                    "query": "How to change theme"
                },
                {
                    "question": "How do I save a post?",
                    "content_type": "saved posts",
                    "query": "How to save posts"
                }
            ]
            
            # Create response dictionary with proper structure
            response = {
                "type": "greeting_with_faqs", 
                "content": {
                    "greeting": random.choice(greeting_responses),
                    "faqs": faqs
                }
            }
            
            # Double-check for valid JSON serialization
            try:
                import json
                # Test if the response is JSON serializable
                json.dumps(response)
            except Exception as e:
                print(f"Error serializing greeting response: {e}")
                # Fallback to a simpler response
                response = {
                    "type": "message",
                    "content": random.choice(greeting_responses)
                }
            
            self.sessions[session_id].append({"role": "assistant", "content": response})
            return response

        what_is_patterns = [
        "what is", "what's", "tell me about", "explain", 
        "describe", "define", "overview of"
        ]
    
        # Detect content type in query
        content_type = detect_content_type(user_input)

        # Special handling for "what is" queries about content types
        if content_type != "none":
            user_input_lower = user_input.lower()
    
            # Check if the query contains a "what is" pattern
            if any(pattern in user_input_lower for pattern in what_is_patterns):
                # Directly provide the explanation from content_explanations
                explanation = self.content_explanations.get(content_type, 
                    f"Here's information about {content_type}.")
        
                # Create a response that suggests the guide and asks user
                response = {
                    "type": "content_explanation_with_guide_prompt",
                    "content": {
                        "explanation": explanation,
                        "content_type": content_type,
                        "prompt": f"Would you like to see the step-by-step guide for creating a {content_type.upper()}?"
                    }
                }
        
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
                
        if self._is_user_search_query(user_input):
            response = {
                "type": "message", 
                "content": "I'm here to help with Bigshorts features. I cannot access user data or find specific profiles. What would you like to know about creating content?"
            }
            self.sessions[session_id].append({"role": "assistant", "content": response})
            return response
    
        # Handle help or guidance requests
        if any(help_term in user_input.lower() for help_term in ["help", "guide", "what can you do", "features", "capabilities", "show me"]):
            if "content types" in user_input.lower() or "features" in user_input.lower() or "guides" in user_input.lower():
                # Provide a categorized overview of content types
                categories = {
                    "Content Creation": ["shot", "snip", "ssup", "collab", "flix"],
                    "Content Editing": ["editing a shot", "editing a ssup", "editing a snip", "editing a flix", "interactive snip"],
                    "Profile Management": ["edit profile", "multiple accounts", "account overview", "change password", "block/unblock user"],
                    "Content Management": ["store draft", "delete post", "edit post", "saved posts", "post insights", "create a playlist"],
                    "App Settings": ["notification", "change theme", "feedback", "invite friends", "report", "hide/unhide users"]
                }
            
                category_response = "Here are the Bigshorts features I can help you with:\n\n"
                for category, features in categories.items():
                    category_response += f"**{category}**\n"
                    category_response += ", ".join([f.upper() for f in features]) + "\n\n"
                
                category_response += "Ask me about any specific feature to learn more!"
            
                response = {"type": "message", "content": category_response}
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
        
        # Handle trending content requests FIRST (before content type detection)
        if any(keyword in user_input.lower() for keyword in ["trending", "popular", "discover", "recommended"]):
            if "snips" in user_input.lower() or "videos" in user_input.lower() or "video" in user_input.lower():
                response = suggest_trending_content("snips")
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
            elif "creators" in user_input.lower() or "users" in user_input.lower() or "people" in user_input.lower():
                response = suggest_trending_content("creators")
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
            elif "shots" in user_input.lower() or "photos" in user_input.lower() or "pictures" in user_input.lower():
                response = suggest_trending_content("shots")
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
            else:
                response = suggest_trending_content("all")
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response

        # Detect content type in query
        content_type = detect_content_type(user_input)


        if content_type == "none" and "bigshorts" in user_input.lower():
            # Generate a generic response about Bigshorts rather than defaulting to a specific guide
            generic_response = {
                "type": "message",
                "content": "I see you're asking about Bigshorts! I can help you with creating content (SHOT, SNIP, SSUP, FLIX), managing your account, using platform features, or troubleshooting issues. What specific aspect of Bigshorts would you like to know more about?"
            }
            self.sessions[session_id].append({"role": "assistant", "content": generic_response})
            return generic_response

        if user_input.startswith("FAQ:"):
            # Extract the content type from the FAQ selection format "FAQ: content_type"
            try:
                selected_content_type = user_input.split("FAQ:")[1].strip()
                
                # If it's an issue, handle it as an issue
                if any(issue_type in selected_content_type.lower() for issue_type in ALLOWED_ISSUE_TYPES):
                    issue_type = next((issue for issue in ALLOWED_ISSUE_TYPES if issue in selected_content_type.lower()), None)
                    if issue_type:
                        response = {"type": "issue", "content": handle_common_issues(issue_type)}
                        self.sessions[session_id].append({"role": "assistant", "content": response})
                        return response
                
                # Otherwise treat it as a content guide request
                response = content_creation_guide(selected_content_type)
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
            except Exception as e:
                print(f"Error handling FAQ selection: {str(e)}")
                # If something goes wrong, fallback to regular processing
                pass


        def get_natural_content_phrasing(content_type: str) -> str:
            """Returns a more natural way to phrase a content type in a suggestion"""
    
            # Dictionary mapping content types to natural phrases
            natural_phrasing = {
                "shot": "create a SHOT",
                "snip": "create a SNIP",
                "ssup": "create a SSUP",
                "collab": "collaborate with other users",
                "editing a shot": "edit your SHOT",
                "invite friends": "invite your friends",
                "feedback": "give feedback",
                "multiple accounts": "manage multiple accounts",
                "account overview": "check your account overview",
                "store draft": "store a draft",
                "change password": "change your password",
                "notification": "manage notifications",
                "change theme": "change the app theme",
                "report": "report content",
                "moment": "create a Moment",
                "delete post": "delete a post",
                "post insights": "view post insights",
                "saved posts": "manage saved posts",
                "edit profile": "edit your profile",
                "edit post": "edit a post",
                "block/unblock user": "block or unblock a user",
                "hide/unhide users": "hide or unhide users",
                "messages": "send messages",
                "discovery": "discover new content",
                "editing a ssup": "edit a SSUP",
                "interactive snip": "create an interactive SNIP",
                "flix": "create a FLIX",
                "create a playlist": "create a playlist",
                "editing a flix": "edit a FLIX",
                "editing a snip": "edit a SNIP"
            }
    
            # Get the natural phrasing or create a fallback
            std_content_type = content_type.lower()
            return natural_phrasing.get(std_content_type, f"use {content_type}")

        # Handle content-specific queries
        if content_type != "none":
            # For general inquiries about content types without action verbs
            action_verbs = ["create", "make", "how to", "guide", "tutorial", "steps", "post", "share", "upload", "show", "explain"]
            if not any(x in user_input.lower() for x in action_verbs):
                # Simple content type inquiry - just the name or "what is X"
                basic_inquiries = [ct for ct in ALLOWED_CONTENT_TYPES]
                what_is_patterns = [f"what is a {ct}" for ct in ALLOWED_CONTENT_TYPES] + [f"what's a {ct}" for ct in ALLOWED_CONTENT_TYPES]
                tell_me_patterns = [f"tell me about {ct}" for ct in ALLOWED_CONTENT_TYPES]
                show_me_patterns = [f"show me {ct}" for ct in ALLOWED_CONTENT_TYPES]
    
                if (user_input.lower().strip() in basic_inquiries or 
                    any(pattern in user_input.lower() for pattern in what_is_patterns + tell_me_patterns + show_me_patterns)):
                    guide = content_creation_guide(content_type)
                    explanation = self.content_explanations.get(content_type, f"Here's information about {content_type}.")
                    self.sessions[session_id].append({"role": "assistant", "content": f"{explanation} Let me show you the guide:"})
                    return guide

                # Get natural phrasing for the content type
                natural_phrase = get_natural_content_phrasing(content_type)
        
                suggestion_response = {
                    "type": "suggestion",
                    "content": f"It looks like you're interested in {content_type}. Would you like me to show you how to {natural_phrase}? Reply 'yes' or ask 'how to {natural_phrase}'."
                }
                self.sessions[session_id].append({"role": "assistant", "content": suggestion_response})
                return suggestion_response
        
            # If it includes action verbs, provide the content guide
            if any(x in user_input.lower() for x in action_verbs):
                response = content_creation_guide(content_type)
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response

        # Handle "yes" replies to suggestions about content creation
        if user_input.lower().strip() in ["yes", "yeah", "sure", "ok", "okay"]:
            try:
                # Check previous assistant message for content-related suggestions
                if len(self.sessions[session_id]) >= 2:
                    prev_message = self.sessions[session_id][-2]
                    if prev_message["role"] == "assistant":
                        prev_content = prev_message["content"]
                
                        # For debugging
                        print(f"DEBUG - Previous message content type: {type(prev_content)}")
                        print(f"DEBUG - Previous message content: {prev_content}")
                
                        # Handle content_explanation_with_guide_prompt type
                        if isinstance(prev_content, dict) and prev_content.get("type") == "content_explanation_with_guide_prompt":
                            content_type = prev_content.get("content", {}).get("content_type")
                            if content_type:
                                response = content_creation_guide(content_type)
                                self.sessions[session_id].append({"role": "assistant", "content": response})
                                return response
                
                        # Check for "Would you like to see the step-by-step guide" in any dict type
                        elif isinstance(prev_content, dict):
                            prompt_text = ""
                    
                            # Try to extract text from different possible formats
                            if "content" in prev_content and isinstance(prev_content["content"], str):
                                prompt_text = prev_content["content"]
                            elif "content" in prev_content and isinstance(prev_content["content"], dict):
                                # Extract from nested content objects
                                for key, value in prev_content["content"].items():
                                    if isinstance(value, str) and "step-by-step guide" in value:
                                        prompt_text = value
                                        break
                    
                            # If we found text with the prompt
                            if "step-by-step guide" in prompt_text:
                                # Find which content type was mentioned
                                for content_type in ALLOWED_CONTENT_TYPES:
                                    if content_type.upper() in prompt_text:
                                        response = content_creation_guide(content_type)
                                        self.sessions[session_id].append({"role": "assistant", "content": response})
                                        return response
                
                        # Handle suggestion type
                        elif isinstance(prev_content, dict) and prev_content.get("type") == "suggestion":
                            suggestion_text = prev_content.get("content", "")
                            # Extract content type from suggestion text
                            for content_type in ALLOWED_CONTENT_TYPES:
                                if content_type.lower() in suggestion_text.lower():
                                    response = content_creation_guide(content_type)
                                    self.sessions[session_id].append({"role": "assistant", "content": response})
                                    return response
                
                        # String content with content type
                        elif isinstance(prev_content, str):
                            # Find the mentioned content type
                            for content_type in ALLOWED_CONTENT_TYPES:
                                if content_type.lower() in prev_content.lower() or content_type.upper() in prev_content:
                                    response = content_creation_guide(content_type)
                                    self.sessions[session_id].append({"role": "assistant", "content": response})
                                    return response
        
                # If no specific content identified, provide a helpful response
                response = {"type": "message", "content": "I'd be happy to help! What specifically would you like guidance on? You can ask about SHOT, SNIP, SSUP, FLIX, or any other Bigshorts feature."}
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
        
            except Exception as e:
                # Log the exception for debugging
                print(f"Exception in 'yes' handler: {str(e)}")
        
                # Provide a fallback response
                response = {"type": "message", "content": "I'd be happy to help with Bigshorts features! Could you please specify which content type you're interested in? For example, SHOT, SNIP, SSUP, FLIX, or Collab?"}
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response

        # Check for off-topic queries
        if self._is_off_topic(user_input):
            response = get_off_topic_response()
            self.sessions[session_id].append({"role": "assistant", "content": response})
            return {"type": "message", "content": response}

        # Handle issues, ideas, and platform sections
        if any(term in user_input.lower() for term in ["problem", "issue", "help with", "trouble", "can't", "doesn't work", "not working", "fix"]):
            issue = self._extract_issue(user_input)
            if issue != "unknown":
                response = {"type": "issue", "content": handle_common_issues(issue)}
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response

        if "snip ideas" in user_input.lower() or "interactive ideas" in user_input.lower() or "ideas for snip" in user_input.lower():
            response = {"type": "idea", "content": generate_interactive_video_ideas()}
            self.sessions[session_id].append({"role": "assistant", "content": response})
            return response

        # Check for platform section questions
        for section in ALLOWED_PLATFORM_SECTIONS:
            if section in user_input.lower():
                response = {"type": "guide", "content": platform_guide(section)}
                self.sessions[session_id].append({"role": "assistant", "content": response})
                return response
            
        # Use the LLM for other queries (with 50% chance to add trending content)
        try:
            llm_response = self.generate_llm_response(user_input)
            self.sessions[session_id].append({"role": "assistant", "content": llm_response})
        
            # 50% chance to add trending content suggestions
            if random.random() < 0.5:
                trending_suggestions = suggest_trending_content("all")
                return {
                    "type": "combined",
                    "content": {
                        "message": llm_response,
                        "trending": trending_suggestions["content"]
                    }
                }
            else:
                return {"type": "message", "content": llm_response}
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            error_response = {
                "type": "error",
                "content": "I'm sorry, I couldn't process that request. Can I help you with creating SHOT, SNIP, SSUP, FLIX, or Collab content? Or would you like guidance on other features like editing, moments, or playlists?"
            }
            self.sessions[session_id].append({"role": "assistant", "content": error_response})
            return error_response
            
    def get_conversation_history(self, session_id: str = None) -> List[Dict[str, str]]:
        """Return the conversation history for a specific session"""
        if session_id is None:
            session_id = "default"
            
        return self.sessions.get(session_id, [])


# Test/demo code
def run_chatbot():
    """Run the chatbot in an interactive loop"""
    # Check if model exists in the models directory
    model_path = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run the setup code first to download the model, or update the path.")
        return
    
    try:
        chatbot = BigshortsChatbot(model_path)
        
        print("Bigshorts Assistant: Hi! I'm your Bigshorts assistant. I can help with creating SHOT, SNIP, SSUP, or Collab content. How can I assist you today?")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("Bigshorts Assistant: Thanks for chatting! Have a great day!")
                    break
                    
                response = chatbot.process_query(user_input)
                
                # Handle different response types
                if isinstance(response, dict):
                    if response.get("type") == "message":
                        print(f"Bigshorts Assistant: {response.get('content')}")
                    elif response.get("type") == "content_guide":
                        print(f"Bigshorts Assistant: Here's a guide for {response.get('content', {}).get('title', 'content creation')}!")
                        print("(Visual guide displayed)")
                    elif response.get("type") == "combined":
                        print(f"Bigshorts Assistant: {response.get('content', {}).get('message', '')}")
                        print("(Trending content suggestions displayed)")
                    elif response.get("type") == "suggestion_buttons":
                        print(f"Bigshorts Assistant: {response.get('content', {}).get('message', '')}")
                        buttons = response.get('content', {}).get('buttons', [])
                        for button in buttons:
                            print(f"- {button.get('text', '')}")
                    elif response.get("type") == "suggestion":
                        print(f"Bigshorts Assistant: {response.get('content', '')}")
                    elif response.get("type") == "issue":
                        print(f"Bigshorts Assistant: {response.get('content', '')}")
                    elif response.get("type") == "idea":
                        print(f"Bigshorts Assistant: {response.get('content', '')}")
                    elif response.get("type") == "error":
                        print(f"Bigshorts Assistant: {response.get('content', '')}")
                    elif response.get("type") == "content_explanation":
                        explanation = response.get('content', {}).get('explanation', '')
                        print(f"Bigshorts Assistant: {explanation}")
                        print("\nWould you like to see a detailed step-by-step guide for creating this content?")
                    elif response.get("type") == "content_explanation_with_guide_prompt":
                        explanation = response.get('content', {}).get('explanation', '')
                        prompt = response.get('content', {}).get('prompt', '')
                        print(f"Bigshorts Assistant: {explanation}")
                        print(f"\n{prompt}")
                    else:
                        print(f"Bigshorts Assistant: {response}")
                else:
                    print(f"Bigshorts Assistant: {response}")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Bigshorts Assistant: I'm sorry, I ran into a technical issue. How else can I help with Bigshorts features?")
    
    except Exception as e:
        print(f"Fatal error initializing chatbot: {str(e)}")
        print("Please check your model path and dependencies.")


if __name__ == "__main__":
    run_chatbot()
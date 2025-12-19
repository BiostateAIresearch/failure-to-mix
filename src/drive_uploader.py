"""
Google Drive integration for uploading results.

Usage in Google Colab:
    from google.colab import auth
    auth.authenticate_user()
    
    uploader = DriveUploader()
    uploader.upload_file("results.csv", folder_id="...")
"""
import os
from typing import Optional, List
from pathlib import Path


class DriveUploader:
    """Upload files to Google Drive."""
    
    def __init__(self, authenticate: bool = True):
        """
        Initialize Drive uploader.
        
        Args:
            authenticate: Whether to authenticate (set False if already done)
        """
        self.drive = None
        self.sheets = None
        self._initialized = False
        
        if authenticate:
            self._init_services()
    
    def _init_services(self):
        """Initialize Google Drive and Sheets services."""
        try:
            from googleapiclient.discovery import build
            
            self.drive = build('drive', 'v3')
            self.sheets = build('sheets', 'v4')
            self._initialized = True
        except Exception as e:
            print(f"⚠️ Could not initialize Google services: {e}")
            print("Make sure to run: from google.colab import auth; auth.authenticate_user()")
    
    def find_spreadsheet(self, name_keyword: str) -> Optional[dict]:
        """
        Find a spreadsheet by name keyword.
        
        Args:
            name_keyword: Keyword to search in spreadsheet names
            
        Returns:
            Dict with id, name, parents or None
        """
        if not self._initialized:
            return None
        
        try:
            page_token = None
            files = []
            
            while True:
                resp = self.drive.files().list(
                    q="mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
                    fields="nextPageToken, files(id,name,modifiedTime,parents)",
                    pageToken=page_token
                ).execute()
                
                files.extend(resp.get('files', []))
                page_token = resp.get('nextPageToken')
                if not page_token:
                    break
            
            # Find matching files
            candidates = [
                f for f in files 
                if name_keyword.lower() in f['name'].lower()
            ]
            
            if not candidates:
                return None
            
            # Return most recently modified
            return sorted(candidates, key=lambda f: f['modifiedTime'], reverse=True)[0]
            
        except Exception as e:
            print(f"⚠️ Error finding spreadsheet: {e}")
            return None
    
    def create_folder(self, name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Create a folder in Drive.
        
        Args:
            name: Folder name
            parent_id: Parent folder ID
            
        Returns:
            Created folder ID or None
        """
        if not self._initialized:
            return None
        
        try:
            metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder"
            }
            if parent_id:
                metadata["parents"] = [parent_id]
            
            folder = self.drive.files().create(
                body=metadata,
                fields="id"
            ).execute()
            
            return folder.get("id")
            
        except Exception as e:
            print(f"⚠️ Error creating folder: {e}")
            return None
    
    def upload_file(
        self,
        local_path: str,
        folder_id: Optional[str] = None,
        name: Optional[str] = None
    ) -> bool:
        """
        Upload a file to Drive.
        
        Args:
            local_path: Path to local file
            folder_id: Target folder ID
            name: Override filename
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            from googleapiclient.http import MediaFileUpload
            
            file_name = name or os.path.basename(local_path)
            metadata = {"name": file_name}
            if folder_id:
                metadata["parents"] = [folder_id]
            
            media = MediaFileUpload(local_path, resumable=False)
            self.drive.files().create(
                body=metadata,
                media_body=media
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error uploading file: {e}")
            return False
    
    def upload_folder(self, local_folder: str, drive_folder_id: str) -> int:
        """
        Upload all files in a folder to Drive.
        
        Args:
            local_folder: Local folder path
            drive_folder_id: Target Drive folder ID
            
        Returns:
            Number of files uploaded
        """
        count = 0
        for root, _, files in os.walk(local_folder):
            for name in files:
                file_path = os.path.join(root, name)
                if self.upload_file(file_path, drive_folder_id):
                    count += 1
                    print(f"✅ Uploaded: {name}")
        return count
    
    def read_sheet(
        self,
        spreadsheet_id: str,
        sheet_name: str,
        range_str: str = "A1:ZZ10000"
    ) -> List[List[str]]:
        """
        Read data from a Google Sheet.
        
        Args:
            spreadsheet_id: Spreadsheet ID
            sheet_name: Worksheet name
            range_str: Cell range to read
            
        Returns:
            List of rows (each row is list of cell values)
        """
        if not self._initialized:
            return []
        
        try:
            range_full = f"{sheet_name}!{range_str}"
            result = self.sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_full,
                majorDimension="ROWS"
            ).execute()
            
            return result.get("values", [])
            
        except Exception as e:
            print(f"⚠️ Error reading sheet: {e}")
            return []
    
    def write_sheet(
        self,
        spreadsheet_id: str,
        sheet_name: str,
        data: List[List],
        start_cell: str = "A1"
    ) -> bool:
        """
        Write data to a Google Sheet.
        
        Args:
            spreadsheet_id: Spreadsheet ID
            sheet_name: Worksheet name
            data: List of rows to write
            start_cell: Starting cell
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            range_str = f"{sheet_name}!{start_cell}"
            
            # Convert data to strings
            str_data = [[str(cell) for cell in row] for row in data]
            
            self.sheets.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_str,
                valueInputOption="RAW",
                body={"values": str_data}
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error writing sheet: {e}")
            return False
    
    def create_sheet(self, spreadsheet_id: str, sheet_name: str) -> bool:
        """
        Create a new worksheet in a spreadsheet.
        
        Args:
            spreadsheet_id: Spreadsheet ID
            sheet_name: Name for new worksheet
            
        Returns:
            True if successful
        """
        if not self._initialized:
            return False
        
        try:
            self.sheets.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [{
                        "addSheet": {
                            "properties": {"title": sheet_name}
                        }
                    }]
                }
            ).execute()
            return True
            
        except Exception as e:
            print(f"⚠️ Error creating sheet: {e}")
            return False

# DeepFace GUI with Database Integration

A comprehensive face analysis application with SQLite database support for storing and managing face analysis results, user data, and verification history.

## Features

### 🎯 Core Functionality
- **Face Verification**: Compare two faces and determine if they belong to the same person
- **Face Analysis**: Analyze faces to determine age, gender, emotion, and ethnicity
- **Dual Image Display**: Shows both images side-by-side during verification
- **Progress Tracking**: Real-time progress bars and status updates
- **Rich Result Display**: Color-coded, structured results with detailed information

### 💾 Database Features
- **User Management**: Track analyses by user
- **Analysis History**: Store all face analysis results
- **Verification History**: Keep track of face comparison operations
- **Face Embeddings**: Store face embeddings for similarity search
- **Data Export**: Export results and database to JSON format
- **Search Functionality**: Find similar faces in the database

### 🎨 Enhanced UI
- **Modern Design**: Professional interface with styled components
- **Responsive Layout**: Adaptive display modes for different operations
- **Progress Indicators**: Visual feedback during processing
- **Error Handling**: Comprehensive error display with suggestions

## Database Schema

### Tables

#### Users
- Store user information and track their analyses
- Fields: id, name, email, created_at, updated_at

#### Face Analyses
- Store face analysis results (both verification and analysis)
- Fields: id, user_id, image_path, analysis_type, result_data, confidence_score, processing_time, model_used, detector_used, created_at

#### Face Embeddings
- Store face embeddings for similarity comparison
- Fields: id, analysis_id, embedding_data, face_location, created_at

#### Verification History
- Store face verification comparisons
- Fields: id, image1_id, image2_id, similarity_score, verified, threshold_used, model_used, detector_used, processing_time, created_at

## Usage

### Basic Operation

1. **Set Current User** (optional): Use File → Set Current User to associate analyses with a user
2. **Load Images**: Click "Load Image 1" and optionally "Load Image 2" for verification
3. **Select Function**: Choose between "Verify Faces" or "Analyze Face"
4. **Execute Analysis**: Click "Execute Analysis" to run the selected operation
5. **View Results**: Results are displayed with rich formatting and automatically saved to database

### Database Management

1. **Open Database Manager**: Use File → Database Manager to access the full database interface
2. **View Statistics**: Dashboard shows database usage statistics
3. **Manage Users**: Add, view, and manage users in the Users tab
4. **Review History**: Check analysis and verification history
5. **Search Similar Faces**: Use the Search tab to find similar faces
6. **Export Data**: Export database or results to JSON files

### Database Manager Features

#### Dashboard Tab
- Real-time database statistics
- Recent activity overview
- Database size and usage metrics

#### Users Tab
- Add new users
- View all registered users
- User analysis history

#### Analyses Tab
- View all face analyses
- Filter by user or analysis type
- Export individual results

#### Verifications Tab
- Review verification history
- See similarity scores and results
- Track verification accuracy

#### Search Tab
- Upload a face image
- Find similar faces in database
- Adjustable similarity thresholds

#### Settings Tab
- Export entire database
- Database optimization
- Clear database (with confirmation)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python deepface_gui.py
```

3. (Optional) Open database manager:
```bash
python database_manager.py
```

## Database Configuration

The database is automatically created as `face_database.db` in the application directory. No additional configuration is required.

### Database Methods

#### Core Database Operations
- `add_user(name, email)`: Add a new user
- `save_analysis(user_id, image_path, analysis_type, result_data, ...)`: Save analysis results
- `save_verification(image1_id, image2_id, similarity_score, ...)`: Save verification results
- `get_verification_history(limit)`: Get recent verifications
- `search_similar_faces(target_embedding, threshold, limit)`: Find similar faces

#### Utility Methods
- `get_database_stats()`: Get database statistics
- `export_data(export_path, data_type)`: Export data to JSON
- `clear_database(confirm)`: Clear all data (requires confirmation)

## File Structure

```
face/
├── deepface_gui.py          # Main GUI application
├── database_manager.py      # Database management interface
├── database.py             # Database core functionality
├── requirements.txt        # Python dependencies
└── face_database.db        # SQLite database (created automatically)
```

## Dependencies

- PyQt5: GUI framework
- DeepFace: Face analysis library
- SQLite3: Database (built-in Python module)
- NumPy: Numerical computations
- JSON: Data serialization

## API Integration

The database system is designed to be easily extensible and can be integrated with other applications or services through the FaceDatabase class.

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Ensure write permissions in the application directory
2. **Image Loading Issues**: Verify image formats (JPG, PNG supported)
3. **Memory Issues**: Large databases can be optimized using the maintenance tools
4. **Permission Errors**: Run as administrator if database creation fails

### Database Maintenance

- Use the Settings tab to optimize database performance
- Regular exports help backup important data
- Monitor database size in the Dashboard tab

## Contributing

The database system is modular and can be extended with additional features such as:
- Face embedding clustering
- Advanced search algorithms
- Data visualization
- API endpoints for web integration

## License

This project is open source and available for educational and commercial use.

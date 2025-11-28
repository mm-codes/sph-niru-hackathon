from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.core.files.storage import default_storage
import os
from .models import AudioAnalysis, TextAnalysis
from .analyzers import AudioAnalyzer, TextAnalyzer
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, IsAuthenticatedOrReadOnly

def landing_page(request):
    return render(request, 'app/landing.html')

def dashboard(request):
    return render(request, 'app/dashboard.html')

def profile(request):
    return render(request, 'app/profile.html')

def analysis(request):
    return render(request, 'app/analysis.html')

def reports(request):
    return render(request, 'app/reports.html')

# Initialize analyzers (in production, use singleton pattern)
audio_analyzer = AudioAnalyzer()
text_analyzer = TextAnalyzer()

class AudioAnalysisView(APIView):
    permission_classes = [IsAuthenticatedOrReadOnly]

    def post(self, request, *args, **kwargs):
        """
        Analyze audio file for deepfake detection
        """
        if 'audio' not in request.FILES:
            return Response(
                {'error': 'No audio file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        audio_file = request.FILES['audio']

        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        if file_ext not in allowed_extensions:
            return Response(
                {'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Save temporarily
            file_path = default_storage.save(f'temp/{audio_file.name}', audio_file)
            full_path = default_storage.path(file_path)

            # Analyze
            results = audio_analyzer.analyze(full_path)

            # Save to database
            analysis = AudioAnalysis.objects.create(
                user=request.user if request.user.is_authenticated else None,
                audio_file=audio_file,
                is_deepfake=results['is_deepfake'],
                confidence_score=results['confidence'],
                analysis_details=results
            )

            # Clean up temp file
            default_storage.delete(file_path)

            return Response({
                'id': str(analysis.id),
                'is_deepfake': results['is_deepfake'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'analysis_type': results['analysis_type'],
                'created_at': analysis.created_at
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class TextAnalysisView(APIView):
    permission_classes = [IsAuthenticatedOrReadOnly]

    def post(self, request, *args, **kwargs):
        """
        Analyze text for AI generation detection
        """
        text = request.data.get('text', '').strip()

        if not text:
            return Response(
                {'error': 'No text provided'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(text) < 50:
            return Response(
                {'error': 'Text must be at least 50 characters long for accurate analysis'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Analyze
            results = text_analyzer.analyze(text)

            # Save to database
            analysis = TextAnalysis.objects.create(
                user=request.user if request.user.is_authenticated else None,
                text_content=text[:1000],  # Store first 1000 chars
                is_ai_generated=results['is_ai_generated'],
                confidence_score=results['confidence'],
                analysis_details=results
            )

            return Response({
                'id': str(analysis.id),
                'is_ai_generated': results['is_ai_generated'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities'],
                'text_length': results['text_length'],
                'word_count': results['word_count'],
                'analysis_type': results['analysis_type'],
                'created_at': analysis.created_at
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AnalysisHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """Get user's analysis history"""
        audio_analyses = AudioAnalysis.objects.filter(user=request.user).order_by('-created_at')[:10]
        text_analyses = TextAnalysis.objects.filter(user=request.user).order_by('-created_at')[:10]

        return Response({
            'audio_analyses': [{
                'id': str(a.id),
                'is_deepfake': a.is_deepfake,
                'confidence': a.confidence_score,
                'created_at': a.created_at
            } for a in audio_analyses],
            'text_analyses': [{
                'id': str(t.id),
                'is_ai_generated': t.is_ai_generated,
                'confidence': t.confidence_score,
                'created_at': t.created_at
            } for t in text_analyses]
        })

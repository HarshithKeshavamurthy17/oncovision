# Portfolio Integration Guide

This guide helps you integrate the OncoVision demo into your portfolio website.

## Quick Demo Links

After deployment, you'll have a live demo URL. Here are integration options:

### Option 1: Direct Link Button

```html
<a href="https://your-demo-url.streamlit.app" 
   target="_blank" 
   class="demo-button">
   🚀 Try Live Demo
</a>
```

### Option 2: Iframe Embed

```html
<div class="demo-container">
    <iframe 
        src="https://your-demo-url.streamlit.app" 
        width="100%" 
        height="800px" 
        frameborder="0"
        allowfullscreen>
    </iframe>
</div>
```

### Option 3: Screenshot with Link

```html
<div class="project-demo">
    <img src="demo-screenshot.png" alt="OncoVision Demo">
    <a href="https://your-demo-url.streamlit.app" 
       target="_blank" 
       class="demo-link">
       View Live Demo →
    </a>
</div>
```

## Portfolio Section Template

### HTML Example

```html
<section class="project" id="oncovision">
    <div class="project-header">
        <h2>OncoVision - Breast Ultrasound Segmentation</h2>
        <div class="tech-stack">
            <span class="badge">PyTorch</span>
            <span class="badge">U-Net</span>
            <span class="badge">Deep Learning</span>
            <span class="badge">Medical Imaging</span>
        </div>
    </div>
    
    <div class="project-content">
        <div class="project-description">
            <p>
                OncoVision is an AI-powered tool for automated segmentation 
                of breast ultrasound images. The system uses a U-Net architecture 
                with ResNet50 encoder to perform multi-class segmentation, 
                distinguishing between benign tumors, malignant tumors, and 
                background tissue.
            </p>
            
            <h3>Key Features:</h3>
            <ul>
                <li>Multi-class segmentation (Background, Benign, Malignant)</li>
                <li>Real-time predictions with visual overlay</li>
                <li>Trained on BUSI dataset with 624 image-mask pairs</li>
                <li>Combined Dice-Focal Loss for imbalanced datasets</li>
                <li>Comprehensive data augmentation pipeline</li>
            </ul>
            
            <h3>Technologies:</h3>
            <ul>
                <li>PyTorch, Segmentation Models PyTorch</li>
                <li>Albumentations for data augmentation</li>
                <li>Streamlit for web interface</li>
                <li>TensorBoard for training visualization</li>
            </ul>
        </div>
        
        <div class="project-demo">
            <h3>Live Demo</h3>
            <a href="https://your-demo-url.streamlit.app" 
               target="_blank" 
               class="demo-button">
                🚀 Try Live Demo
            </a>
            <p class="demo-note">
                Upload a breast ultrasound image to see real-time segmentation results
            </p>
        </div>
        
        <div class="project-links">
            <a href="https://github.com/HarshithKeshavamurthy17/oncovision" 
               target="_blank" 
               class="github-link">
                📁 View on GitHub
            </a>
        </div>
    </div>
</section>
```

### CSS Styling Example

```css
.demo-button {
    display: inline-block;
    padding: 12px 24px;
    background-color: #1f77b4;
    color: white;
    text-decoration: none;
    border-radius: 5px;
    font-weight: bold;
    transition: background-color 0.3s;
}

.demo-button:hover {
    background-color: #1565a0;
}

.demo-container {
    margin: 20px 0;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}

.tech-stack {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.badge {
    padding: 4px 12px;
    background-color: #f0f0f0;
    border-radius: 12px;
    font-size: 0.9em;
}
```

## React/Next.js Example

```jsx
import Link from 'next/link';

export default function OncoVisionProject() {
    return (
        <section className="project">
            <h2>OncoVision - Breast Ultrasound Segmentation</h2>
            
            <p>
                AI-powered tool for automated segmentation of breast 
                ultrasound images using deep learning.
            </p>
            
            <div className="tech-stack">
                <span>PyTorch</span>
                <span>U-Net</span>
                <span>Deep Learning</span>
                <span>Medical Imaging</span>
            </div>
            
            <div className="project-links">
                <Link 
                    href="https://your-demo-url.streamlit.app"
                    target="_blank"
                    className="demo-button"
                >
                    🚀 Try Live Demo
                </Link>
                <Link 
                    href="https://github.com/HarshithKeshavamurthy17/oncovision"
                    target="_blank"
                    className="github-link"
                >
                    📁 View Code
                </Link>
            </div>
        </section>
    );
}
```

## Screenshots for Portfolio

Take screenshots of:
1. **Demo Interface** - Show the upload and results
2. **Segmentation Results** - Show before/after comparisons
3. **Training Metrics** - TensorBoard graphs (optional)
4. **Code Structure** - Clean project organization

## Demo Video

Record a short demo video (1-2 minutes):
1. Show the interface
2. Upload a sample image
3. Display results
4. Explain key features

Host on YouTube/Vimeo and embed in portfolio.

## Portfolio Description Template

```markdown
## OncoVision - Breast Ultrasound Image Segmentation

**Project Type:** Deep Learning, Medical Imaging, Computer Vision

**Duration:** [Your duration]

**Technologies:** PyTorch, U-Net, ResNet50, Streamlit, TensorBoard

### Overview
OncoVision is an AI-powered tool for automated segmentation of breast 
ultrasound images. The system uses deep learning to identify and classify 
different tissue types, helping in early detection of breast cancer.

### Key Achievements
- Achieved [X]% Dice score on validation set
- Implemented multi-class segmentation (Background, Benign, Malignant)
- Built interactive web demo using Streamlit
- Applied advanced data augmentation techniques
- Handled class imbalance with weighted loss functions

### Technical Highlights
- Architecture: U-Net with ResNet50 encoder
- Loss Function: Combined Dice-Focal Loss
- Data Augmentation: 12+ augmentation techniques
- Training: 50 epochs with early stopping
- Evaluation: Dice Score, IoU, Precision, Recall

### Demo
[Live Demo Link] | [GitHub Repository]
```

## Adding to Resume/CV

```
OncoVision - Breast Ultrasound Segmentation
• Developed deep learning model using U-Net architecture with ResNet50 encoder
• Implemented multi-class segmentation achieving X% Dice score
• Created interactive web demo using Streamlit for real-time predictions
• Applied data augmentation and class balancing techniques
• Technologies: PyTorch, TensorBoard, Streamlit, OpenCV
```

## Social Media Posts

### LinkedIn Post Template

```
🔬 Excited to share OncoVision - my latest project on AI-powered breast 
ultrasound image segmentation!

Using deep learning with U-Net architecture, I built a system that can 
automatically segment breast ultrasound images into benign tumors, 
malignant tumors, and background tissue.

Key features:
✅ Multi-class segmentation
✅ Real-time predictions
✅ Interactive web demo
✅ Comprehensive evaluation metrics

Try it live: [demo link]
View code: [GitHub link]

#DeepLearning #MedicalAI #ComputerVision #PyTorch #MachineLearning
```

## Tips for Success

1. **Keep it Updated**: Regularly update your demo with improvements
2. **Document Well**: Clear README and documentation
3. **Show Results**: Include metrics and visualizations
4. **Be Responsive**: Make sure demo works on mobile too
5. **Monitor Usage**: Track demo usage if possible
6. **Add Disclaimers**: Medical tools need proper disclaimers

## Support

If you need help with deployment or integration, check:
- [DEPLOYMENT.md](./DEPLOYMENT.md) for deployment instructions
- [README.md](./README.md) for project documentation
- GitHub Issues for troubleshooting





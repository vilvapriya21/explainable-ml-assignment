"""Pydantic schemas for the model evaluation API."""

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Represent model-service health information."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str


class PredictRequest(BaseModel):
    """Represent all breast-cancer benchmark input features."""

    model_config = ConfigDict(populate_by_name=True)

    mean_radius: float = Field(..., alias="mean radius", description="Mean radius.")
    mean_texture: float = Field(..., alias="mean texture", description="Mean texture.")
    mean_perimeter: float = Field(..., alias="mean perimeter", description="Mean perimeter.")
    mean_area: float = Field(..., alias="mean area", description="Mean area.")
    mean_smoothness: float = Field(..., alias="mean smoothness", description="Mean smoothness.")
    mean_compactness: float = Field(..., alias="mean compactness", description="Mean compactness.")
    mean_concavity: float = Field(..., alias="mean concavity", description="Mean concavity.")
    mean_concave_points: float = Field(..., alias="mean concave points", description="Mean concave points.")
    mean_symmetry: float = Field(..., alias="mean symmetry", description="Mean symmetry.")
    mean_fractal_dimension: float = Field(..., alias="mean fractal dimension", description="Mean fractal dimension.")
    radius_error: float = Field(..., alias="radius error", description="Radius error.")
    texture_error: float = Field(..., alias="texture error", description="Texture error.")
    perimeter_error: float = Field(..., alias="perimeter error", description="Perimeter error.")
    area_error: float = Field(..., alias="area error", description="Area error.")
    smoothness_error: float = Field(..., alias="smoothness error", description="Smoothness error.")
    compactness_error: float = Field(..., alias="compactness error", description="Compactness error.")
    concavity_error: float = Field(..., alias="concavity error", description="Concavity error.")
    concave_points_error: float = Field(..., alias="concave points error", description="Concave points error.")
    symmetry_error: float = Field(..., alias="symmetry error", description="Symmetry error.")
    fractal_dimension_error: float = Field(..., alias="fractal dimension error", description="Fractal dimension error.")
    worst_radius: float = Field(..., alias="worst radius", description="Worst radius.")
    worst_texture: float = Field(..., alias="worst texture", description="Worst texture.")
    worst_perimeter: float = Field(..., alias="worst perimeter", description="Worst perimeter.")
    worst_area: float = Field(..., alias="worst area", description="Worst area.")
    worst_smoothness: float = Field(..., alias="worst smoothness", description="Worst smoothness.")
    worst_compactness: float = Field(..., alias="worst compactness", description="Worst compactness.")
    worst_concavity: float = Field(..., alias="worst concavity", description="Worst concavity.")
    worst_concave_points: float = Field(..., alias="worst concave points", description="Worst concave points.")
    worst_symmetry: float = Field(..., alias="worst symmetry", description="Worst symmetry.")
    worst_fractal_dimension: float = Field(..., alias="worst fractal dimension", description="Worst fractal dimension.")


class PredictResponse(BaseModel):
    """Represent one classifier prediction."""

    model_config = ConfigDict(protected_namespaces=())

    predicted_class: int
    class_label: str
    probability: float
    model_version: str


class FeatureContribution(BaseModel):
    """Represent one feature's local contribution."""

    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    """Represent a prediction and its strongest local contributions."""

    predicted_class: int
    class_label: str
    probability: float
    top_features: list[FeatureContribution]


class ModelReviewResponse(BaseModel):
    """Represent persisted agent review markdown state."""

    content: str
    generated: bool

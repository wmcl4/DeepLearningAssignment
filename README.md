Write me

Solar Panel Segmentation from Aerial Imagery using U-Net

## 1. Introduction & Motivation

Accurately mapping the location and extent of rooftop and ground-mounted solar photovoltaic (PV) installations is valuable for a range of applications: utilities and grid operators need this data to plan for distributed energy resource integration, policymakers use it to track renewable energy adoption at a regional level, and researchers use it to estimate solar potential and identify underutilized rooftops. Manually annotating solar panels across large aerial or satellite image collections is slow and does not scale to national or continental coverage.

This project addresses this problem as a **binary semantic segmentation task**: given an RGB aerial image tile, predict a pixel-wise mask indicating which pixels belong to a solar panel versus background (roofs, vegetation, roads, open ground, etc.). We use a convolutional encoder-decoder network (U-Net) trained on labeled aerial imagery to automate this detection process.

## 2. Dataset & Architecture

























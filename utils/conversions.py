def convert_px_to_m(pixel_dist,ref_height_in_m,ref_height_in_px):
    return (pixel_dist*ref_height_in_m)/ref_height_in_px

def convert_m_to_px(m,ref_height_in_m,ref_height_in_px):
    return (m*ref_height_in_px)/ref_height_in_m

